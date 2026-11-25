"""The real pipeline from Python: one source -> target pair of the paper's translator, a
document, questions, a full translated cache and one compressed cache per question, answered by
the target.

NEEDS: a CUDA GPU (80 GB for 14B pairs at long documents; the source is parked on the CPU
outside its prefill), the released checkpoint and the catalog models from Hugging Face.
``--device cpu`` / ``mps`` runs the eager reference path (fast-inference level 0) and works,
but real 8-14B models make it slow and memory-hungry; use ``examples/quickstart_cpu.py`` to try
the flow on a laptop.

    # the released checkpoint (downloads only this pair's files)
    uv run python examples/encode_and_answer.py --ratio 128 \\
        --source qwen3-14b --target llama31-8b --document contract.txt \\
        --question "Who signed the agreement?"
    # or a local folder (milestone, split or export: detected automatically)
    ... --checkpoint ckpt/ --compressor 128=ckpt/compressor-128x/compressor-step-002400.pt
    # answer with vLLM instead (a subprocess; needs the vllm extra, see README)
    ... --engine vllm

The same thing from the command line: ``examples/encode_and_answer_cli.sh``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speculative_encoding import checkpoints  # noqa: E402
from speculative_encoding.api import SpeculativeEncoder, answer_vllm  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="hf",
        help="checkpoint folder, or hf (default: the released checkpoint)")
    parser.add_argument("--source", default="qwen3-14b")
    parser.add_argument("--target", default="llama31-8b")
    parser.add_argument("--document", required=True, help="text file")
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument(
        "--ratio", type=float, action="append", default=[],
        help="with --checkpoint hf: compression ratios to fetch (128 is released)")
    parser.add_argument(
        "--compressor", action="append", default=[], help="ratio=path of a local compressor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="encoded")
    parser.add_argument("--engine", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--vllm-python", help="--engine vllm: interpreter for vLLM (default: this one)")
    args = parser.parse_args()

    compressors = {float(r): p for r, p in (item.split("=", 1) for item in args.compressor)}
    checkpoint = args.checkpoint
    if checkpoint == "hf":
        found = checkpoints.resolve(args.source, args.target, ratios=args.ratio)
        checkpoint, compressors = found.checkpoint, {**found.compressors, **compressors}
        print(
            f"checkpoint {checkpoint} "
            f"({len(found.files)} files at {checkpoints.REPO_ID}@{found.revision[:8]})")

    enc = SpeculativeEncoder.from_pretrained(
        checkpoint, args.source, args.target, compressors=compressors, device=args.device)
    questions = [enc.question(text) for text in args.question]
    doc = enc.encode(
        Path(args.document).read_text(), questions=questions, ratios=sorted(compressors))
    print(f"full cache: {doc.full.doc_tokens} document tokens, {doc.full.bytes / 2**20:.1f} MiB")
    for (row_id, ratio), item in sorted(
        doc.compressed.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))
    ):
        print(
            f"{ratio:g}x cache for {row_id}: {item.meta['slots']} tokens kept, "
            f"{item.bytes / 2**20:.2f} MiB")
    doc.save(args.out, Path(args.document).stem)

    if args.engine == "hf":
        for item in [doc.full, *doc.compressed.values()]:
            for question, answer in zip(item.questions, enc.answer_hf(item)):
                print(f"[{item.arm}] {question['text']} -> {answer['text']!r}")
    else:
        enc.release()  # vLLM takes the GPU in its own process
        for record in answer_vllm(args.out, python=args.vllm_python, max_model_len=40960):
            print(f"[{record['name']}] {record.get('row_id')} -> {record['translated']['text']!r}")


if __name__ == "__main__":
    main()
