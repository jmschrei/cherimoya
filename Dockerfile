# syntax=docker/dockerfile:1
# droast ignore=DF007 reason="added a .dockerignore file"
FROM ghcr.io/astral-sh/uv:0.11.30-python3.12-trixie-slim@sha256:193af66bebd2668fd3cdc75176690e7e0956182cb5ff88dea156d278a8b16fa6 AS builder

WORKDIR /work
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS main

LABEL org.opencontainers.image.title=cherimoya
LABEL org.opencontainers.image.source=https://github.com/jmschrei/cherimoya

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
CMD ["python3", "-c", "from cherimoya import Cherimoya; print('cherimoya imported OK')"]
