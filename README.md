# speculative-encoding

A source LLM reads a document once; a small shared translator turns its per-layer hidden
states into a target LLM's KV cache; an optional query-aware compressor shrinks that
cache 16x / 64x / 128x; vLLM (or Hugging Face) answers from the injected cache. One
translator covers seven models (Qwen3 14B, Qwen3.5 9B, Llama 3.1 8B, Gemma 3 12B, Gemma 4
12B, Ministral 3 8B and 14B) in all 42 directions.

## Quick start

Try the whole flow on a laptop first, in seconds and with no downloads (tiny random
models and a random translator through the real API, so the answers are noise):

```bash
uv sync
uv run python examples/quickstart_cpu.py   # --device mps on Apple silicon
```

On a Linux CUDA machine, one environment holds the translator and vLLM 0.24 (Torch 2.11,
transformers 5.23). vLLM 0.24 reads two transformers symbols that 5.23 removed; a small
patch loaded at interpreter start restores them:

```bash
uv sync --extra vllm --extra fast
uv run python -m speculative_encoding.serving.transformers_compat --install
uv run python -m speculative_encoding.serving.transformers_compat --check
```

Qwen 3.5 (as source or target) also needs its linear-attention kernels (pure Triton):

```bash
uv pip install einops fla-core==0.6.0 flash-linear-attention==0.6.0
```

Without `causal-conv1d`, Transformers runs Qwen 3.5's short convolution in PyTorch: correct, a
little slower. It has no wheel for this Torch (CUDA 13); building it needs a CUDA 13 `nvcc`.

The released checkpoint is
[`RampPublic/speculative-encoding`](https://huggingface.co/RampPublic/speculative-encoding)
on the Hugging Face Hub: the translator (`translator/`, all seven models) and the 128x
compressor (`compressor-128x/`). `--checkpoint hf` (the CLI's default) or
`checkpoints.resolve(source, target, ratios=[128])` (Python) downloads only one pair's
files. Any local checkpoint folder works too; its format (milestone, split or export) is
detected. The translator holds every model's weights; split a local copy once so a pair
loads only its own files (about 1.5 GB in bf16):

```bash
uv run python -m speculative_encoding.cli split path/to/translator ckpt/split
```

## Run on Baseten

`run_on_baseten.py` runs the whole pipeline on one Baseten GPU: it builds the environment
from `uv.lock`, downloads the released checkpoint, encodes a document
(`examples/sample_document.txt` by default) and answers a question with vLLM; the answers are in the job's logs. It needs a Baseten secret holding a
Hugging Face token with access to the gated models (Llama 3.1, Gemma):

```bash
HF_SECRET=<secret name> uvx truss train push run_on_baseten.py --job-name speculative-encoding-demo
```

`SOURCE`, `TARGET`, `DOCUMENT`, `DOCUMENT_URL`, `QUESTION` and `COMPRESSOR` set at push time are
passed to the job (`examples/baseten_run.sh`). `COMPRESSOR=128` also answers from a 128x
cache; use it with a long document (`DOCUMENT_URL`), since 128x of the 2k-token sample is about 16 tokens.

## Command line

```bash
uv run python -m speculative_encoding.cli encode \
  --source qwen3-14b --target llama31-8b --document contract.txt \
  --question "Who signed the agreement?" --compressor 128 \
  --out encoded/ --name contract
uv run python -m speculative_encoding.cli answer --folder encoded/ --engine vllm
uv run python -m speculative_encoding.cli answer --folder encoded/ --engine hf  # no vLLM
```

`encode` writes the full translated cache and one compressed cache per question and
ratio; `answer` answers every saved item. `--help` on each command lists the options
(metric, sampling, device, chunk size).

## Python API

```python
from speculative_encoding import checkpoints
from speculative_encoding.api import SpeculativeEncoder, answer_vllm

found = checkpoints.resolve("qwen3-14b", "llama31-8b", ratios=[128])  # or a local folder
enc = SpeculativeEncoder.from_pretrained(
    found.checkpoint, source="qwen3-14b", target="llama31-8b", compressors=found.compressors)
q = enc.question("Who signed the agreement?")
doc = enc.encode(open("contract.txt").read(), questions=[q], ratios=[128])
print(enc.answer_hf(doc.full, [q])[0]["text"])  # Hugging Face, same process
doc.save("encoded/", "contract")
enc.release()                                   # free the GPU for vLLM
answers = answer_vllm("encoded/", max_model_len=40960)          # vLLM in a subprocess
```

- **Questions.** `enc.question(text, metric="free", max_tokens=1024,
  sampling={"temperature": 0.7, "top_p": 0.9, "seed": 0})` asks a free-form question
  with no MCQ scaffolding; the sampling options travel with the saved question to both
  engines. `metric="nll", answer=reference` scores a reference text instead. It returns
  a `prompts.Question` (`.to_dict()` is the saved form; every function taking questions
  accepts either); `prompts.render_question(tokenizer, text)` renders one for any
  tokenizer, e.g. a relay's first reader.
  `hf_engine.stream(...)` yields tokens as they are produced and
  `hf_engine.continue_turn(...)` continues a conversation on the cached document;
  with `stream(..., keep_cache=True)` the last event's `result` also carries the cache it
  generated with, so a relay can capture the answer's rows from it
  (`handoff.capture_rows`, see `handoff`).
- **What the compressor kept.** Compressed items carry `kept_positions` and
  `kept_char_spans` (character spans into the string you passed) in `meta`;
  `encode(..., return_scores=True)` adds the per-token selection scores and
  `doc_char_spans` (one span per scored token, to paint the whole document),
  `return_scores=k` the top `k`; `meta["timings"]` has the selection, compression and
  cache-write seconds.
- **Controls.** `enc.answer_hf(item, native=True)` answers with the target reading the
  document itself (for a compressed item, the whole document): the baseline to show next to
  the translated or compressed answer. `cli answer --native` does the same with vLLM.
- **One source, many targets.** `MultiTargetEncoder.from_pretrained(ckpt, source,
  targets=[...])` loads the translator and source once, reads each document once and
  returns `{target: EncodedDocument}`. Questions are passed as text and rendered per
  target. See `examples/fanout.py`.
- **Chains.** `SpeculativeEncoder.from_pretrained_pairs(ckpt, [(A, B), (B, C)])` returns
  `{(source, target): encoder}` over one translator and one model pool: each model is
  loaded once and stays resident (B is the first hop's target and the second's source).
- **Token-level handoff.** `enc.encode_ids(ids, source_rows=rows)` encodes source token
  ids (e.g. a model's own generation) with rows from `handoff.capture_rows`. The
  translator was trained on document-prefill rows; rows captured in other contexts are
  plumbed exactly but their translation quality is unmeasured.
- **Saved items.** `Encoded.load(folder, name)` reads back what `save` wrote
  (`EncodedDocument.save(folder, name)`'s full cache: `Encoded.load(folder, name)` too);
  `Encoded.load_folder(folder)` reads every item.
- **Prefix algebra.** `prefix_ops` concatenates separately encoded documents, extends a
  served prefix with native tokens and truncates.

## Smaller GPUs

- The source model stays on the CPU except during its prefill.
- `enc.source_rows(text)` computes the source's hidden states once (`torch.save` them);
  `encode(..., source_rows=rows)` then needs no source model.
- `enc.encode_memory(text)` returns the translator's encoder output, which depends only on
  the source and is 5-7x smaller than the hidden states (`.save(path)` /
  `EncodedMemory.load(path)`). `encode(memory=mem)` on any encoder with that source skips
  the source prefill and the encoder; `SpeculativeEncoder(..., load_source=False)` never
  loads the source model.
- `chunk=16384` (translator tokens per call) fits Gemma pairs at 32k on an 80 GB GPU. An
  out-of-memory error during translation suggests a smaller chunk; output is identical.
- `MultiTargetEncoder(..., offload_targets=True)` keeps one target model on the GPU at a
  time.
- `device="cpu"` or `"mps"` runs the eager reference path (fast-inference level 0, no
  bf16 autocast); the fast path (levels 1-6, Triton kernels) needs CUDA.

## Layout

Read in this order:

| File | What |
|---|---|
| `examples/quickstart_cpu.py` | the whole flow on CPU with tiny random models |
| `speculative_encoding/api.py` | `SpeculativeEncoder`, `MultiTargetEncoder`, `Encoded`, `answer_vllm` |
| `speculative_encoding/cli.py` | `split`, `encode`, `answer` |
| `speculative_encoding/encode.py` | the encode stage: translator direction, cache writers, KVzip selection, stored encoder output |
| `speculative_encoding/prompts.py` | token alignment, chat rendering and samples, grading |
| `speculative_encoding/models.py` | the model catalog, loading models, source hidden states |
| `speculative_encoding/checkpoints.py` | loading any checkpoint format, splitting, saving, downloading one pair |
| `speculative_encoding/handoff.py`, `prefix_ops.py` | encoding from token ids (relays); prefix algebra and key rotation |
| `speculative_encoding/settings.py` | library switches; CUDA vs CPU/MPS |
| `speculative_encoding/translator/` | `model` (the translator), `compressor`, `hf` (each model's HF interface), `fast_inference` and `kernels` (CUDA fast path) |
| `speculative_encoding/serving/` | `hf_engine` (answer with HF), `vllm_engine` and `vllm_connector` (answer with vLLM), `transformers_compat` |

## Provenance

The serving path is the code behind the paper's long-document numbers, reshaped step by
step. Every step is checked against that reference: bit-exact on CPU (translator,
compressor, selection, prompts for 18 tokenizers, checkpoint loading) and end to end on
H100 for all seven models.
