#!/usr/bin/env bash
# Usage examples:
# ./docker-shell.sh # open bash in the container
# ./docker-shell.sh jupyter lab --LabApp.token='' --ip=0.0.0.0
# IMAGE_NAME=model-dev ./docker-shell.sh python -c "import torch; print(torch.__version__)"


set -euo pipefail


IMAGE_NAME=${IMAGE_NAME:-indie-model-uv}
CONTAINER_NAME=${CONTAINER_NAME:-${IMAGE_NAME}-$(basename "${PWD}")}
JUPYTER_PORT=${JUPYTER_PORT:-8888}
API_PORT=${API_PORT:-8000}


# Add GPU if available (NVIDIA runtime)
GPU_FLAG=()
if command -v nvidia-smi >/dev/null 2>&1; then
GPU_FLAG=(--gpus all)
fi


# Build if the image is missing
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
echo "[docker-shell] Building image '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" -f Dockerfile .
fi


# Host HF cache (optional)
HF_CACHE_HOST=${HF_CACHE_HOST:-$HOME/.cache/huggingface}
mkdir -p "$HF_CACHE_HOST"


DOCKER_OPTS=(
--rm -it
"${GPU_FLAG[@]}"
--name "$CONTAINER_NAME"
-v "$PWD":/app
-v "$HF_CACHE_HOST":/app/.cache/huggingface
-w /app
-p ${JUPYTER_PORT}:8888
-p ${API_PORT}:8000
--user "$(id -u):$(id -g)"
-e HF_HOME=/app/.cache/huggingface
-e TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
)


if [ $# -eq 0 ]; then
exec docker run "${DOCKER_OPTS[@]}" "$IMAGE_NAME" /bin/bash -lc "source /app/.venv/bin/activate && exec bash"
else
exec docker run "${DOCKER_OPTS[@]}" "$IMAGE_NAME" /bin/bash -lc "source /app/.venv/bin/activate && exec $*"
fi
