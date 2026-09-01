FROM node:22-alpine
WORKDIR /app
COPY server.js ./
COPY public ./public
EXPOSE 8000
ENV PORT=8000
CMD ["node", "server.js"]
