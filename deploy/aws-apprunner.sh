#!/usr/bin/env bash
# Deploy LatexGen to AWS App Runner: one always-on container serving the static
# site, the server-model proxy (any OpenAI-compatible API) and the Tab API /
# mesh relay. Chosen over Lambda because the relay is stateful (long-polls,
# in-memory queues) and a single instance keeps that valid; CloudFront can be
# put in front later for caching the model weights.
#
# Usage:
#   AWS_PROFILE=demo OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
#   OPENAI_MODEL=qwen/qwen3-8b OPENAI_REFINE_MODEL=qwen/qwen3-32b \
#   OPENAI_API_KEY_SECRET_ARN=arn:aws:secretsmanager:...:secret:latexgen/openai \
#   deploy/aws-apprunner.sh
# The API key is read by App Runner from Secrets Manager; it never sits in the
# image or the frontend.
set -euo pipefail
EXPECTED_ACCOUNT="${EXPECTED_ACCOUNT:-830940976371}"
REGION="${AWS_REGION:-us-east-1}"
APP="latexgen"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
[[ "$ACCOUNT" == "$EXPECTED_ACCOUNT" ]] || { echo "REFUSING: account $ACCOUNT != $EXPECTED_ACCOUNT" >&2; exit 1; }
: "${OPENAI_BASE_URL:?set OPENAI_BASE_URL (e.g. https://openrouter.ai/api/v1)}"
: "${OPENAI_MODEL:?set OPENAI_MODEL}"
OPENAI_REFINE_MODEL="${OPENAI_REFINE_MODEL:-$OPENAI_MODEL}"

echo "==> ECR"
REPO_URI=$(aws ecr describe-repositories --repository-names "$APP" --region "$REGION" --query 'repositories[0].repositoryUri' --output text 2>/dev/null \
  || aws ecr create-repository --repository-name "$APP" --region "$REGION" --image-scanning-configuration scanOnPush=true --query 'repository.repositoryUri' --output text)
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO_URI%%/*}" >/dev/null
TAG="$(git -C "$DIR" rev-parse --short HEAD 2>/dev/null || date +%s)"
docker build --platform linux/amd64 -t "$REPO_URI:$TAG" -t "$REPO_URI:latest" "$DIR"
docker push "$REPO_URI:$TAG" >/dev/null && docker push "$REPO_URI:latest" >/dev/null
echo "pushed $REPO_URI:$TAG"

echo "==> App Runner access role (pull from ECR)"
ROLE="$APP-apprunner-ecr"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"build.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
  sleep 10
fi
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)
INSTANCE_ROLE_ARN=""
if [[ -n "${OPENAI_API_KEY_SECRET_ARN:-}" ]]; then
  IROLE="$APP-apprunner-instance"
  if ! aws iam get-role --role-name "$IROLE" >/dev/null 2>&1; then
    aws iam create-role --role-name "$IROLE" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"tasks.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam put-role-policy --role-name "$IROLE" --policy-name read-openai-secret --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"secretsmanager:GetSecretValue\"],\"Resource\":\"$OPENAI_API_KEY_SECRET_ARN\"}]}"
    sleep 10
  fi
  INSTANCE_ROLE_ARN=$(aws iam get-role --role-name "$IROLE" --query Role.Arn --output text)
fi

ENV_JSON="{\"NODE_ENV\":\"production\",\"TRUST_PROXY\":\"1\",\"HSTS\":\"1\",\"OLLAMA_DISABLED\":\"1\",\"OPENAI_BASE_URL\":\"$OPENAI_BASE_URL\",\"OPENAI_MODEL\":\"$OPENAI_MODEL\",\"OPENAI_REFINE_MODEL\":\"$OPENAI_REFINE_MODEL\",\"PORT\":\"8000\"}"
SECRETS_JSON="{}"; [[ -n "${OPENAI_API_KEY_SECRET_ARN:-}" ]] && SECRETS_JSON="{\"OPENAI_API_KEY\":\"$OPENAI_API_KEY_SECRET_ARN\"}"
SRC=$(cat <<JSON
{"ImageRepository":{"ImageIdentifier":"$REPO_URI:$TAG","ImageRepositoryType":"ECR",
  "ImageConfiguration":{"Port":"8000","RuntimeEnvironmentVariables":$ENV_JSON,"RuntimeEnvironmentSecrets":$SECRETS_JSON}},
 "AutoDeploymentsEnabled":false,
 "AuthenticationConfiguration":{"AccessRoleArn":"$ROLE_ARN"}}
JSON
)
INSTANCE_CFG="{\"Cpu\":\"0.25 vCPU\",\"Memory\":\"0.5 GB\"$([[ -n "$INSTANCE_ROLE_ARN" ]] && echo ",\"InstanceRoleArn\":\"$INSTANCE_ROLE_ARN\"")}"
HEALTH='{"Protocol":"HTTP","Path":"/api/health","Interval":10,"Timeout":5,"HealthyThreshold":1,"UnhealthyThreshold":5}'

echo "==> App Runner service"
SVC_ARN=$(aws apprunner list-services --region "$REGION" --query "ServiceSummaryList[?ServiceName=='$APP'].ServiceArn | [0]" --output text)
if [[ "$SVC_ARN" == "None" || -z "$SVC_ARN" ]]; then
  SVC_ARN=$(aws apprunner create-service --region "$REGION" --service-name "$APP" \
    --source-configuration "$SRC" --instance-configuration "$INSTANCE_CFG" --health-check-configuration "$HEALTH" \
    --auto-scaling-configuration-arn "$(aws apprunner list-auto-scaling-configurations --region "$REGION" --query 'AutoScalingConfigurationSummaryList[?AutoScalingConfigurationName==`DefaultConfiguration`].AutoScalingConfigurationArn | [0]' --output text)" \
    --query Service.ServiceArn --output text)
else
  aws apprunner update-service --region "$REGION" --service-arn "$SVC_ARN" --source-configuration "$SRC" --instance-configuration "$INSTANCE_CFG" >/dev/null
fi
# NOTE: the relay is single-instance state; keep MaxSize=1 in the autoscaling config.
URL=$(aws apprunner describe-service --region "$REGION" --service-arn "$SVC_ARN" --query Service.ServiceUrl --output text)
echo ""
echo "==> deploying; check status with: aws apprunner describe-service --service-arn $SVC_ARN --query Service.Status"
echo "==> URL: https://$URL"
