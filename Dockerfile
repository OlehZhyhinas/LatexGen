# LatexGen: static frontend + relay + server-model proxy. Zero npm dependencies.
# The frontend is the server build of scripts/build-static.mjs: model weights
# stream from the CDN, so the image carries the app, not 2.2 GB of weights.
FROM node:22-alpine AS build
WORKDIR /src
COPY scripts/build-static.mjs scripts/
COPY public ./public
RUN SERVER_BUILD=1 node scripts/build-static.mjs

FROM node:22-alpine
ENV NODE_ENV=production PORT=8000
WORKDIR /app
COPY --chown=node:node server.js ./
COPY --from=build --chown=node:node /src/dist ./public
USER node
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8000/api/health >/dev/null || exit 1
CMD ["node", "server.js"]
