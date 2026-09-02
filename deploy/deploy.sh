#!/usr/bin/env bash
# Deploy LatexGen to AWS: S3 (private) + CloudFront + Lambda escalation proxy.
# Usage: AWS_PROFILE=demo ./deploy.sh
# Expects the profile to point at the personal DemoAccount (830940976371).
set -euo pipefail

EXPECTED_ACCOUNT="830940976371"
REGION="${AWS_REGION:-us-east-1}"
APP="latexgen"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if [[ "$ACCOUNT" != "$EXPECTED_ACCOUNT" ]]; then
  echo "REFUSING: credentials are for account $ACCOUNT, expected DemoAccount $EXPECTED_ACCOUNT" >&2
  exit 1
fi
echo "==> deploying to account $ACCOUNT ($REGION)"

BUCKET="$APP-site-$ACCOUNT"

# ---- 1. S3 bucket (private; CloudFront-only access via OAC) ----
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  aws s3api put-public-access-block --bucket "$BUCKET" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi

echo "==> syncing static site"
aws s3 sync "$DIR/public/" "s3://$BUCKET/" --delete \
  --exclude "bench*" \
  --cache-control "public,max-age=300"
# model weights and vendored libraries are immutable — long cache
for prefix in models vendor; do
  aws s3 cp "s3://$BUCKET/$prefix/" "s3://$BUCKET/$prefix/" --recursive \
    --metadata-directive REPLACE --cache-control "public,max-age=31536000,immutable" >/dev/null
done
# S3 guesses content types poorly for these; WebAssembly streaming needs application/wasm
aws s3 cp "s3://$BUCKET/vendor/" "s3://$BUCKET/vendor/" --recursive --exclude "*" --include "*.wasm" \
  --metadata-directive REPLACE --content-type application/wasm --cache-control "public,max-age=31536000,immutable" >/dev/null
aws s3 cp "s3://$BUCKET/vendor/" "s3://$BUCKET/vendor/" --recursive --exclude "*" --include "*.mjs" \
  --metadata-directive REPLACE --content-type "text/javascript; charset=utf-8" --cache-control "public,max-age=31536000,immutable" >/dev/null

# ---- 2. Lambda ----
ROLE_NAME="$APP-lambda-role"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "waiting for role propagation…"; sleep 10
fi
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)

echo "==> packaging lambda"
(cd "$DIR/deploy/lambda" && zip -q -j /tmp/$APP-lambda.zip index.mjs)

if aws lambda get-function --function-name "$APP-api" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$APP-api" --region "$REGION" \
    --zip-file fileb:///tmp/$APP-lambda.zip >/dev/null
else
  aws lambda create-function --function-name "$APP-api" --region "$REGION" \
    --runtime nodejs20.x --handler index.handler --role "$ROLE_ARN" \
    --zip-file fileb:///tmp/$APP-lambda.zip --timeout 60 --memory-size 256 >/dev/null
fi
# NOTE: no API key is set — escalation stays disabled until you run:
#   aws lambda update-function-configuration --function-name latexgen-api \
#     --environment 'Variables={OPENAI_BASE_URL=https://openrouter.ai/api/v1,OPENAI_API_KEY=...,OPENAI_MODEL=...,OPENAI_REFINE_MODEL=...}'

FURL=$(aws lambda create-function-url-config --function-name "$APP-api" --region "$REGION" \
  --auth-type NONE --query FunctionUrl --output text 2>/dev/null || \
  aws lambda get-function-url-config --function-name "$APP-api" --region "$REGION" \
  --query FunctionUrl --output text)
aws lambda add-permission --function-name "$APP-api" --region "$REGION" \
  --statement-id public-url --action lambda:InvokeFunctionUrl \
  --principal '*' --function-url-auth-type NONE 2>/dev/null || true
FURL_DOMAIN=$(echo "$FURL" | sed -E 's#https?://([^/]+)/?#\1#')
echo "lambda url: $FURL"

# ---- 3. CloudFront: OAC for S3 + /api/* behavior to Lambda ----
OAC_ID=$(aws cloudfront list-origin-access-controls \
  --query "OriginAccessControlList.Items[?Name=='$APP-oac'].Id | [0]" --output text)
if [[ "$OAC_ID" == "None" || -z "$OAC_ID" ]]; then
  OAC_ID=$(aws cloudfront create-origin-access-control --origin-access-control-config \
    "Name=$APP-oac,OriginAccessControlOriginType=s3,SigningBehavior=always,SigningProtocol=sigv4" \
    --query OriginAccessControl.Id --output text)
fi

EXISTING=$(aws cloudfront list-distributions \
  --query "DistributionList.Items[?Comment=='$APP'].Id | [0]" --output text 2>/dev/null)
if [[ "$EXISTING" != "None" && -n "$EXISTING" ]]; then
  echo "distribution $EXISTING already exists — updating origins is manual; site content already synced."
  DOMAIN=$(aws cloudfront get-distribution --id "$EXISTING" --query Distribution.DomainName --output text)
else
  cat > /tmp/$APP-cf.json <<EOF
{
  "CallerReference": "$APP-$(date +%s)",
  "Comment": "$APP",
  "Enabled": true,
  "DefaultRootObject": "index.html",
  "Origins": { "Quantity": 2, "Items": [
    { "Id": "s3", "DomainName": "$BUCKET.s3.$REGION.amazonaws.com",
      "OriginAccessControlId": "$OAC_ID", "S3OriginConfig": { "OriginAccessIdentity": "" } },
    { "Id": "api", "DomainName": "$FURL_DOMAIN",
      "CustomOriginConfig": { "HTTPPort": 80, "HTTPSPort": 443, "OriginProtocolPolicy": "https-only" } }
  ]},
  "DefaultCacheBehavior": {
    "TargetOriginId": "s3", "ViewerProtocolPolicy": "redirect-to-https",
    "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
    "Compress": true,
    "AllowedMethods": { "Quantity": 2, "Items": ["GET","HEAD"],
      "CachedMethods": { "Quantity": 2, "Items": ["GET","HEAD"] } }
  },
  "CacheBehaviors": { "Quantity": 1, "Items": [
    { "PathPattern": "/api/*", "TargetOriginId": "api",
      "ViewerProtocolPolicy": "https-only",
      "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
      "OriginRequestPolicyId": "b689b0a8-53d0-40ab-baf2-68738e2966ac",
      "AllowedMethods": { "Quantity": 7, "Items": ["GET","HEAD","OPTIONS","PUT","POST","PATCH","DELETE"],
        "CachedMethods": { "Quantity": 2, "Items": ["GET","HEAD"] } } }
  ]}
}
EOF
  DIST=$(aws cloudfront create-distribution --distribution-config file:///tmp/$APP-cf.json \
    --query Distribution --output json)
  DIST_ID=$(echo "$DIST" | python3 -c "import json,sys; print(json.load(sys.stdin)['Id'])")
  DOMAIN=$(echo "$DIST" | python3 -c "import json,sys; print(json.load(sys.stdin)['DomainName'])")
  DIST_ARN=$(echo "$DIST" | python3 -c "import json,sys; print(json.load(sys.stdin)['ARN'])")

  aws s3api put-bucket-policy --bucket "$BUCKET" --policy "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{ \"Effect\": \"Allow\",
      \"Principal\": { \"Service\": \"cloudfront.amazonaws.com\" },
      \"Action\": \"s3:GetObject\", \"Resource\": \"arn:aws:s3:::$BUCKET/*\",
      \"Condition\": { \"StringEquals\": { \"AWS:SourceArn\": \"$DIST_ARN\" } } }]}"
  echo "distribution $DIST_ID creating (takes ~5 min)…"
fi

echo ""
echo "==> DONE. Site: https://$DOMAIN"
echo "    Escalation: disabled until OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL are set on the lambda."
