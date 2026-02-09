FROM debian:bookworm-slim
ARG TARGETARCH
ARG TARGETVARIANT

ARG TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGE=torch==2.6.0
ARG INSTALL_EXTRAS=zeroconf,transformers,sherpa,onnx-asr,speaker,web

# Install faster-whisper
WORKDIR /usr/src

COPY ./pyproject.toml ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    \
    && python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        'setuptools<81' \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url "${TORCH_EXTRA_INDEX_URL}" \
        "${TORCH_PACKAGE}" \
    && printf '%s\n' "${TORCH_PACKAGE}" > /tmp/pip-constraints.txt \
    \
    && CC=gcc .venv/bin/pip3 install --no-cache-dir \
        --constraint /tmp/pip-constraints.txt \
        --extra-index-url https://www.piwheels.org/simple \
        -e ".[${INSTALL_EXTRAS}]" \
    && if echo ",${INSTALL_EXTRAS}," | grep -q ",speaker,"; then \
        .venv/bin/python3 -c "import pkg_resources; import resemblyzer; import webrtcvad"; \
    fi \
    \
    && rm -f /tmp/pip-constraints.txt \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

EXPOSE 10400 8099

ENTRYPOINT ["bash", "docker_run.sh"]
