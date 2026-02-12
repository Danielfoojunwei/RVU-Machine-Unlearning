FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -e .[dev,bench]

CMD ["bash"]
