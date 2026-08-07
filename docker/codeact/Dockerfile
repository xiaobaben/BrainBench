FROM python:3.9-slim

ARG DEBIAN_MIRROR=http://deb.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get \
        -o Acquire::Retries=5 \
        -o Acquire::http::Timeout=120 \
        -o Acquire::https::Timeout=120 \
        update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir \
        --index-url "$PIP_INDEX_URL" \
        --timeout "$PIP_DEFAULT_TIMEOUT" \
        --retries "$PIP_RETRIES" \
        -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock \
    && mkdir -p /input /workspace

WORKDIR /workspace
