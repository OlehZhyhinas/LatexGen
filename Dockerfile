# LatexGen: static frontend + relay + server-model proxy. Zero npm dependencies.
FROM node:22-alpine
ENV NODE_ENV=production PORT=8000
WORKDIR /app
COPY --chown=node:node server.js ./
COPY --chown=node:node public ./public
USER node
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8000/api/health >/dev/null || exit 1
CMD ["node", "server.js"]
