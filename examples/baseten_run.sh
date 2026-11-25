#!/usr/bin/env bash
# The pipeline on one Baseten GPU: what run_on_baseten.py starts. Builds the one environment from uv.lock, downloads
# the released checkpoint, encodes a document with SOURCE, answers QUESTION with TARGET (vLLM) and prints the answers.
#
#   DOCUMENT      a text file in the repo to read (default: examples/sample_document.txt, a short contract)
#   DOCUMENT_URL  or a text file to download instead, cut to DOCUMENT_CHARS characters
#   QUESTION      the question (default: one about the sample)
#   SOURCE, TARGET (default qwen3-14b -> llama31-8b); COMPRESSOR=128 also answers from a 128x cache (for long
#                 documents: the 2k-token sample would keep about 16 tokens)
#   HF_TOKEN      read access to the gated catalog models (Llama, Gemma), from a Baseten secret
set -euo pipefail

SOURCE=${SOURCE:-qwen3-14b}
TARGET=${TARGET:-llama31-8b}
DOCUMENT=${DOCUMENT:-examples/sample_document.txt}
DOCUMENT_CHARS=${DOCUMENT_CHARS:-60000}
QUESTION=${QUESTION:-"Who signed the agreement for each party, and when does it end?"}
OUT=${OUT:-/tmp/encoded}
COMPRESSOR=${COMPRESSOR:-}

python -m pip install -q uv==0.12.1
uv sync --frozen --extra vllm --extra fast
uv run python -m speculative_encoding.serving.transformers_compat --install
uv run python -m speculative_encoding.serving.transformers_compat --check
case "$SOURCE,$TARGET" in *qwen35*)  # Qwen 3.5's linear-attention kernels (README)
  uv pip install einops fla-core==0.6.0 flash-linear-attention==0.6.0 setuptools wheel ninja
  CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --no-build-isolation causal-conv1d==1.7.0 ;;
esac

if [[ -n ${DOCUMENT_URL:-} ]]; then
  uv run python -c 'import sys, urllib.request; text = urllib.request.urlopen(sys.argv[1]).read().decode("utf-8", "replace"); open("/tmp/document.txt", "w").write(text[:int(sys.argv[2])])' \
    "$DOCUMENT_URL" "$DOCUMENT_CHARS"
  DOCUMENT=/tmp/document.txt
fi
uv run python -m speculative_encoding.cli encode --source "$SOURCE" --target "$TARGET" \
  --document "$DOCUMENT" --question "$QUESTION" ${COMPRESSOR:+--compressor "$COMPRESSOR"} --out "$OUT"
uv run python -m speculative_encoding.cli answer --folder "$OUT" --engine vllm
