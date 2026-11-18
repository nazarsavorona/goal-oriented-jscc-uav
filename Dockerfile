FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# OpenCV system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies before copying source (better layer caching)
RUN pip install --no-cache-dir \
        "ultralytics==8.4.47" \
        "einops==0.8.2" \
        "wandb>=0.18,<0.19" \
        "opencv-python-headless>=4.9" \
        "scipy>=1.10" \
        "matplotlib>=3.7" \
        "tqdm" \
        "pillow" \
        "timm>=0.9" \
        "transformers>=4.45" \
        "huggingface_hub>=0.24" \
        "torchmetrics>=1.0" \
        "termcolor" \
        "fvcore" \
        "iopath"

# Copy the package metadata (pyproject.toml names README.md and LICENSE), the source and
# the scripts, then install semvideo as an editable package
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY scripts/ scripts/
RUN pip install --no-cache-dir -e "."
