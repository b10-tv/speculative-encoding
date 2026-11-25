"""Fan-out: read a document once with one source model, answer from several target
families.

NEEDS a CUDA GPU, the released checkpoint (or a local folder) and the catalog models.

``MultiTargetEncoder`` loads the translator and the source once. With ``share="memory"``
the source prefill and the translator's encoder run once per document, and every target
decodes from that stored encoder output; ``offload_targets=True`` keeps one target model
on the GPU at a time:

    uv run python examples/fanout.py --document contract.txt \\
        --question "Who signed the agreement?" --targets llama31-8b,gemma3-12b,ministral3-8b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speculative_encoding import checkpoints  # noqa: E402
from speculative_encoding.api import MultiTargetEncoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="hf",
                        help="checkpoint folder, or hf (default: the released checkpoint)")
    parser.add_argument("--source", default="qwen3-14b")
    parser.add_argument("--targets", default="llama31-8b,gemma3-12b,ministral3-8b")
    parser.add_argument("--document", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    targets, text = args.targets.split(","), Path(args.document).read_text()

    checkpoint = args.checkpoint
    if checkpoint == "hf":
        # Files of one commit share one snapshot folder, which ends up holding every pair.
        revision = None
        for target in targets:
            found = checkpoints.resolve(args.source, target, revision=revision)
            checkpoint, revision = found.checkpoint, found.revision

    enc = MultiTargetEncoder.from_pretrained(checkpoint, args.source, targets,
                                             offload_targets=True)
    docs = enc.encode(text, questions=[args.question], share="memory")
    for target, doc in docs.items():
        answer = enc.encoder(target).answer_hf(doc.full)[0]["text"]
        enc.free(target)
        print(f"{target}: {answer!r}")


if __name__ == "__main__":
    main()
