# syntax=docker/dockerfile:1
# SharedAgents stack: orchestrator (8099) and gateway (8100) on the container's
# loopback, WebUI on 8110. Only the WebUI is published; the launcher refuses
# non-loopback backend binds.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SHAREDAGENTS_HOME=/data \
    SHAREDAGENTS_WEB_HOST=0.0.0.0 \
    # Docker's port publish is the proxy in front of the WebUI; login stays on.
    SHAREDAGENTS_WEB_TRUST_PROXY=1

WORKDIR /src
COPY . .
RUN pip install ".[web]" \
    && useradd --create-home --uid 1000 sharedagents \
    && mkdir /data && chown sharedagents:sharedagents /data

USER sharedagents
WORKDIR /data
VOLUME /data
EXPOSE 8110
# `serve` stops all three components on Ctrl+C.
STOPSIGNAL SIGINT
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD ["sharedagents-serve", "status"]
CMD ["sharedagents-serve"]
