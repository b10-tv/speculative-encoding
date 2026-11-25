"""The whole flow in seconds on a laptop CPU, with no downloads: random weights everywhere, so
the answers are noise.

    uv run python examples/quickstart_cpu.py              # cpu
    uv run python examples/quickstart_cpu.py --device mps # Apple GPU (eager reference path too)

What runs is the real API (``speculative_encoding.api.SpeculativeEncoder``) end to end, with
tiny models injected instead of the paper's checkpoints:

* a tiny random Llama (the source) and a tiny random Qwen3 (the target), each with its own
  small byte-level BPE tokenizer trained here on a paragraph, and a chat template;
* a small randomly initialised ``UnifiedTranslator`` and one random query compressor (ratio 4);
* ``encode``: source prefill -> translator -> the target's KV cache of the document, plus one
  compressed cache for the question; ``answer_hf``: the target answers from each cache with the
  Hugging Face engine (it never reads the document's tokens).

Swap in real checkpoints (``examples/encode_and_answer.py``) and nothing else changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speculative_encoding.api import SpeculativeEncoder  # noqa: E402
from speculative_encoding.translator import compressor as C  # noqa: E402
from speculative_encoding.translator.model import UnifiedConfig  # noqa: E402
from speculative_encoding.translator.hf import ModelInterface  # noqa: E402
from speculative_encoding.translator.model import UnifiedTranslator  # noqa: E402

DOCUMENT = (
    "The Harbor Lease was signed on 3 March 2021 by Mira Okafor for the landlord and by "
    "Tomas Reyes for Bluewater Cargo. The rent is 4,200 euros a month, due on the first "
    "working day. Either side may end the lease with ninety days of written notice. The "
    "tenant keeps the north warehouse and two loading bays.")
QUESTION = "Who signed the lease for Bluewater Cargo?"
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")
SPECIALS = ["<|bos|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]


def tiny_tokenizer(vocab_size: int):
    """A byte-level BPE tokenizer trained on the document and the question (deterministic),
    with a chat template."""

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(
        [DOCUMENT, QUESTION, "You are a helpful assistant. Answer the question."] * 4, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<|bos|>", eos_token="<|end|>", pad_token="<|end|>",
        chat_template=CHAT_TEMPLATE)


def tiny_model(family: str, tokenizer, seed: int):
    import transformers as T

    common = dict(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, num_hidden_layers=3, max_position_embeddings=1024,
        tie_word_embeddings=False)
    config, cls = {
        "llama": (T.LlamaConfig, T.LlamaForCausalLM),
        "qwen3": (T.Qwen3Config, T.Qwen3ForCausalLM),
    }[family]
    torch.manual_seed(seed)
    model = cls(config(**common)).eval().requires_grad_(False)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return ModelInterface(model, family), tokenizer


def randomize_outputs(module):
    """Training zero-initialises the output projections (attention ``out``, the FFN ``down``
    maps, the router, ``memory_state_map``) so a fresh translator starts as a no-op; with
    random weights everywhere else that would leave it blind to the source rows."""

    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.dim() > 1 and not parameter.any():
                parameter.normal_(0.0, 0.05)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or mps")
    parser.add_argument(
        "--ratio", type=float, default=4.0, help="compression ratio of the random compressor")
    args = parser.parse_args()
    started = time.perf_counter()

    source, target = "tiny-llama", "tiny-qwen3"
    models = {
        source: tiny_model("llama", tiny_tokenizer(400), 0),
        target: tiny_model("qwen3", tiny_tokenizer(500), 1),
    }
    # The compressor reads the question through the target's encoder input maps, so the target
    # is a source too.
    sources = {key: interface.source_geometry() for key, (interface, _) in models.items()}
    targets = {target: models[target][0].target_geometry()}
    torch.manual_seed(2)
    translator = UnifiedTranslator(
        targets, source_geometries=sources,
        config=UnifiedConfig(
            width=64, heads=4, ffn=128, experts=4, top_k=2, encoder_blocks=1, decoder_blocks=2,
            window=8))
    randomize_outputs(translator)
    with tempfile.TemporaryDirectory() as folder:
        # the API loads compressors from files, as the paper's are shipped
        path = Path(folder) / "compressor.pt"
        compressor = randomize_outputs(C.SlotCompressor(translator.decoder.blocks, blocks=2))
        torch.save(compressor.state_dict(), path)
        encoder = SpeculativeEncoder(
            translator, source, target, device=args.device, models=models,
            compressors={args.ratio: path})

        question = encoder.question(QUESTION, max_tokens=16)
        doc = encoder.encode(DOCUMENT, questions=[question], ratios=[args.ratio])
        compressed = doc.compressed[(question.row_id, args.ratio)]
        print(
            f"document: {doc.full.doc_tokens} target tokens -> "
            f"full cache {doc.full.bytes / 1024:.1f} KiB; "
            f"{args.ratio:g}x cache keeps {compressed.meta['slots']} tokens "
            f"({compressed.bytes / 1024:.1f} KiB), at characters "
            f"{compressed.meta['kept_char_spans'][:4]}...")
        for name, item in (("full", doc.full), (f"{args.ratio:g}x", compressed)):
            answer = encoder.answer_hf(item, [question])[0]
            print(f"answer from the {name} cache (random weights, so noise): {answer['text']!r}")
        saved = doc.save(Path(folder) / "encoded", "lease")
        print(f"saved {len(saved)} items in the queue format: {[p.name for p in saved]}")
        # The cache must depend on the source's states (other rows for the same tokens change
        # it), and the answer only on the cache (prefix_ids are never read).
        rows = encoder.source_rows(DOCUMENT)
        other = encoder.encode(DOCUMENT, source_rows=rows.roll(1, dims=0)).full.prefix
        ids = list(doc.full.prefix_ids)
        random.Random(0).shuffle(ids)
        shuffled = dataclasses.replace(doc.full, prefix_ids=ids)
        same = encoder.answer_hf(shuffled, [question])[0]["ids"]
        checks = {"cache reads the source": not torch.equal(other, doc.full.prefix),
                  "answer ignores prefix_ids": same == encoder.answer_hf(doc.full)[0]["ids"]}
        print("checks: " + ", ".join(f"{name}: {ok}" for name, ok in checks.items()))
    print(f"done on {args.device} in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
