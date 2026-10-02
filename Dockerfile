# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libcurl4 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY pyproject.toml .
RUN pip3 install --no-cache-dir --break-system-packages \
    "gymnasium>=0.29" \
    "stable-baselines3>=2.3" \
    "torch>=2.0" \
    "numpy>=1.24" \
    "tensorboard>=2.14" \
    "pyyaml>=6.0" \
    "tqdm" "rich" \
    "setuptools<81" \
    "onnx" "onnxscript"

# Copy project
COPY . .
RUN pip3 install --no-cache-dir --break-system-packages .

# Native libs: libwinbolo_gym.so, libSDL3.so.0, libSDL3_ttf.so.0
# These must be present in the build context (project root).
# At runtime they are at /app/ and LD_LIBRARY_PATH includes /app.
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

# Default: run training
ENTRYPOINT ["python3"]
CMD ["scripts/train.py", "--config", "configs/phase0_boat.yaml"]
