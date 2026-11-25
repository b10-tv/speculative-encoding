"""Command line: split checkpoints, encode documents, answer questions.

    python -m speculative_encoding.cli split BUNDLE OUT [--dtype bf16]
    python -m speculative_encoding.cli encode --checkpoint CKPT --source qwen3-14b \\
        --target llama31-8b --document doc.txt --question "Who signed?" \\
        [--question ...] [--compressor 128=path ...] --out encoded/ [--name doc]
    python -m speculative_encoding.cli encode --compressor 128 ...   # the released checkpoint
    python -m speculative_encoding.cli answer --folder encoded/ --engine vllm
    python -m speculative_encoding.cli answer --folder encoded/ --engine hf

``encode`` writes the full translated cache (``<name>-full``) and one compressed cache
per question and compressor ratio; ``answer`` answers every saved item (``--engine
vllm`` in the vLLM environment, ``--engine hf`` in this process with Hugging Face)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _split(args):
    import torch

    from speculative_encoding.checkpoints import split_bundle

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    manifest = split_bundle(args.bundle, args.out, dtype=dtype)
    summary = {key: value for key, value in manifest.items() if key != "files"}
    print(json.dumps(summary, indent=1, default=str))


def _encode(args):
    from speculative_encoding.api import SpeculativeEncoder

    # --compressor: ratio=path, or a bare ratio with --checkpoint hf
    given = [item.partition("=") for item in args.compressor]
    compressors = {float(ratio): path for ratio, _, path in given if path}
    checkpoint = args.checkpoint
    if checkpoint != "hf" and len(compressors) != len(given):
        sys.exit("--compressor: ratio=path (a bare ratio needs --checkpoint hf)")
    if checkpoint == "hf":  # download only this pair's files of the released checkpoint
        from speculative_encoding import checkpoints

        found = checkpoints.resolve(args.source, args.target,
                                    ratios=[float(ratio) for ratio, _, _ in given])
        checkpoint, compressors = found.checkpoint, {**found.compressors, **compressors}
    encoder = SpeculativeEncoder.from_pretrained(
        checkpoint, args.source, args.target, compressors=compressors, device=args.device,
        level=args.level, chunk=args.chunk)
    sampling = None
    if args.temperature:
        sampling = {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed}
    questions = [encoder.question(text, metric=args.metric, max_tokens=args.max_tokens,
                                  sampling=sampling) for text in args.question]
    encoded = encoder.encode(Path(args.document).read_text(), questions=questions,
                             ratios=[float(r) for r in compressors])
    paths = encoded.save(args.out, args.name)
    compressed = {f"{row}@{ratio:g}x": item.bytes
                  for (row, ratio), item in encoded.compressed.items()}
    print(json.dumps({"saved": [str(p) for p in paths], "full_bytes": encoded.full.bytes,
                      "compressed_bytes": compressed}, indent=1))


def _answer(args):
    if args.engine == "vllm":
        from speculative_encoding.api import answer_vllm

        records = answer_vllm(args.folder, python=args.vllm_python, out=args.out,
                              max_model_len=args.max_model_len, native=args.native, gpu=args.gpu)
    else:
        import torch

        from speculative_encoding import models as se_models
        from speculative_encoding.api import Encoded
        from speculative_encoding.models import CATALOG
        from speculative_encoding.serving import hf_engine

        folder = Path(args.folder)
        items = {name: item for name, item in Encoded.load_folder(folder, map_location=args.device).items()
                 if item.questions}
        targets = {item.target for item in items.values()}
        if not targets <= set(CATALOG):
            sys.exit(f"{sorted(targets - set(CATALOG))} not in the catalog")
        records = []
        for target in sorted(targets):
            _, interface, tokenizer = se_models.load_catalog(target, torch.device(args.device))
            for name, item in items.items():
                if item.target != target:
                    continue
                answers = hf_engine.answer_batch(interface, tokenizer, item.prefix,
                                                 item.prefix_ids, item.questions)
                for question, answer in zip(item.questions, answers):
                    keys = ("nll",) if "nll" in answer else ("text", "finish_reason")
                    records.append({"name": name, "row_id": question.get("row_id"),
                                    "translated": {key: answer.get(key) for key in keys}})
        out = Path(args.out) if args.out else folder / "answers-hf.jsonl"
        out.write_text("".join(json.dumps(record) + "\n" for record in records))
    for record in records:
        answer = record["translated"]
        shown = f"nll {answer['nll']}" if "nll" in answer else repr(answer["text"][:200])
        print(f"{record['name']}  {record.get('row_id')}: {shown}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="speculative-encoding")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="split a paper milestone into per-model files")
    split.add_argument("bundle")
    split.add_argument("out")
    split.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")

    encode = sub.add_parser("encode", help="translate (and compress) a document and save it")
    encode.add_argument("--checkpoint", default="hf",
                        help="checkpoint folder (milestone, split or export), or hf (default: "
                        "download this pair's files of the released checkpoint)")
    encode.add_argument("--source", required=True)
    encode.add_argument("--target", required=True)
    encode.add_argument("--document", required=True, help="text file")
    encode.add_argument("--question", action="append", default=[])
    encode.add_argument("--metric", default="f1",
                        help="answer format of the questions (f1: short free text; free: "
                        "ungraded free-form; mcq; number; yesno)")
    encode.add_argument("--max-tokens", type=int,
                        help="answer length limit per question (default: the metric's)")
    encode.add_argument("--temperature", type=float, default=0.0,
                        help="sample answers (0: greedy, the paper's)")
    encode.add_argument("--top-p", type=float, default=1.0)
    encode.add_argument("--seed", type=int)
    encode.add_argument("--compressor", action="append", default=[],
                        help="ratio=path, e.g. 128=compressor-128x/compressor-step-002400.pt "
                        "(with --checkpoint hf: just the ratio; 128 is released)")
    encode.add_argument("--out", required=True)
    encode.add_argument("--name", default="doc")
    encode.add_argument("--device", default="cuda",
                        help="cuda (the paper path), or cpu / mps (eager reference path)")
    encode.add_argument("--level", type=int, default=None,
                        help="fast-inference level (default: 6 on CUDA, 0 elsewhere; "
                        ">0 needs CUDA)")
    encode.add_argument("--chunk", type=int, default=32768,
                        help="translator tokens per call (16384 fits Gemma pairs on 80 GB)")

    answer = sub.add_parser("answer", help="answer every saved item of a folder")
    answer.add_argument("--folder", required=True)
    answer.add_argument("--engine", choices=("vllm", "hf"), default="vllm")
    answer.add_argument("--vllm-python", help="interpreter for vLLM (default: this one)")
    answer.add_argument("--out")
    answer.add_argument("--max-model-len", type=int, default=40960)
    answer.add_argument("--native", action="store_true",
                        help="vLLM: also answer from the real document (needs its ids saved)")
    answer.add_argument("--gpu")
    answer.add_argument("--device", default="cuda")

    args = parser.parse_args(argv)
    {"split": _split, "encode": _encode, "answer": _answer}[args.command](args)


if __name__ == "__main__":
    main()
