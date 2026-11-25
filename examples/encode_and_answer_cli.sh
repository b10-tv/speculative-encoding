#!/usr/bin/env bash
# The real pipeline from the command line. NEEDS a CUDA GPU, the released checkpoint and the catalog models.
#
#   bash examples/encode_and_answer_cli.sh contract.txt "Who signed the agreement?"
#
# CHECKPOINT: hf (default: downloads only this pair's files of RampPublic/speculative-encoding) or a local folder
# (milestone, split or export; detected automatically).
# For a laptop, run examples/quickstart_cpu.py instead (tiny random models, no downloads).
set -euo pipefail

DOCUMENT=${1:?document text file}
QUESTION=${2:?question}
CHECKPOINT=${CHECKPOINT:-hf}
SOURCE=${SOURCE:-qwen3-14b}
TARGET=${TARGET:-llama31-8b}
OUT=${OUT:-encoded}
ENGINE=${ENGINE:-vllm}  # or hf (no vLLM)

if [[ $CHECKPOINT == hf ]]; then
  COMPRESSOR=(--compressor 128)  # fetched from compressor-128x/
else
  COMPRESSOR=(--compressor "128=${COMPRESSOR_128:?set COMPRESSOR_128 to a local compressor-step-002400.pt}")
fi

uv run python -m speculative_encoding.cli encode --checkpoint "$CHECKPOINT" --source "$SOURCE" --target "$TARGET" \
  --document "$DOCUMENT" --question "$QUESTION" "${COMPRESSOR[@]}" --out "$OUT" --name doc

uv run python -m speculative_encoding.cli answer --folder "$OUT" --engine "$ENGINE"
