#!/bin/sh
set -e

MODEL="${EMBEDDER_MODEL:-nomic-embed-text-v1.5.Q4_K_M.gguf}"
MODEL_PATH="/models/${MODEL}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "ERROR: Embedding model not found at ${MODEL_PATH}" >&2
  echo "" >&2
  echo "Recall requires 768-dimensional embeddings (pgvector vector(768))." >&2
  echo "Download nomic-embed-text-v1.5 Q4_K_M into your models directory:" >&2
  echo "" >&2
  echo "  mkdir -p \"\${MODELS_HOST_PATH:-\$HOME/models}\"" >&2
  echo "  curl -L -o \"\${MODELS_HOST_PATH:-\$HOME/models}/nomic-embed-text-v1.5.Q4_K_M.gguf \\" >&2
  echo "    https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf" >&2
  exit 1
fi

# Reuse the same llama.cpp image as the generator but stay on CPU (no /dev/dri).
# llama-server --embeddings exposes OpenAI-compatible POST /v1/embeddings.
exec llama-server \
  -m "${MODEL_PATH}" \
  --embeddings \
  --host 0.0.0.0 \
  --port "${EMBEDDER_PORT:-8081}" \
  -c 8192
