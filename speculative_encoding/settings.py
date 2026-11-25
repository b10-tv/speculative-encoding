"""Library switches (``current()``, ``use()``) and device helpers (CUDA vs CPU/MPS)."""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Mapping
from dataclasses import dataclass
import torch


# ---- Library switches --------------------------------------------------------------------------
# Library switches, in one place.
#
# ``current()`` returns the innermost ``use(...)`` override, else ``Settings.from_env()``
# (``PMT_*`` / ``UT_*`` environment variables), read at the moment each switch is used::
#
#     from speculative_encoding import settings
#     with settings.use(settings.Settings(translate_chunk=16384)):
#         ...
#
# ``tiled_attention`` (also ``fast_inference.set_tiled_attention``), ``fused_glue`` and
# ``glue_tune`` are read once, when ``translator.fast_inference`` is imported. ``HF_TOKEN``
# is read as Hugging Face reads it.

@dataclass(frozen=True)
class Settings:
    # Fast path (levels >= 5): tiled window attention instead of FlashAttention-2 with one
    # sequence per token. Off for Ministral 3 targets in the paper runs (the kernel
    # faulted on some 128k shapes).
    tiled_attention: bool = False  # PMT_FAST_TILED_ATTENTION=1
    # Fast path (levels >= 4): fused norm / residual / gate glue kernels.
    fused_glue: bool = False  # PMT_FAST_FUSED_GLUE=1
    # With fused glue: let Inductor tune its kernels' launch configurations.
    glue_tune: bool = False  # PMT_FAST_GLUE_TUNE=1
    # Eval harness (ShardedTranslator.kv): translate documents longer than this many target
    # tokens in serial shards (0: whole).
    translate_chunk: int = 0  # PMT_TRANSLATE_CHUNK
    # models.source_rows_host: source prefill chunk length in tokens.
    source_chunk: int = 32768  # PMT_SOURCE_CHUNK
    # Eval harness (StreamingHybridWriter): run Hugging Face's whole gated-delta module
    # instead of the fast kernels.
    gdn_full_module: bool = False  # PMT_GDN_FULL_MODULE=1
    # ModelInterface.self_check: bf16 cache tolerance (the paper jobs set 3e-2).
    cache_tolerance: float = 2e-2  # UT_BF16_CACHE_TOLERANCE
    # vLLM connector with tensor parallelism > 1: folder the workers load each prefix from.
    vllm_export_dir: str | None = None  # PMT_VLLM_EXPORT_DIR

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        return cls(
            tiled_attention=env.get("PMT_FAST_TILED_ATTENTION", "0") == "1",
            fused_glue=env.get("PMT_FAST_FUSED_GLUE", "0") == "1",
            glue_tune=env.get("PMT_FAST_GLUE_TUNE", "0") == "1",
            translate_chunk=int(env.get("PMT_TRANSLATE_CHUNK", "0")),
            source_chunk=int(env.get("PMT_SOURCE_CHUNK", "32768")),
            gdn_full_module=env.get("PMT_GDN_FULL_MODULE") == "1",
            cache_tolerance=float(env.get("UT_BF16_CACHE_TOLERANCE", "2e-2")),
            vllm_export_dir=env.get("PMT_VLLM_EXPORT_DIR"),
        )


_OVERRIDE: contextvars.ContextVar[Settings | None] = contextvars.ContextVar(
    "speculative_encoding_settings", default=None)


def current() -> Settings:
    """The innermost ``use(...)``, else the environment as it is now."""

    return _OVERRIDE.get() or Settings.from_env()


@contextlib.contextmanager
def use(settings: Settings):
    token = _OVERRIDE.set(settings)
    try:
        yield settings
    finally:
        _OVERRIDE.reset(token)


# ---- Devices -----------------------------------------------------------------------------------
# Device helpers: the paper path runs on CUDA; CPU and Apple MPS run the eager
# reference path in the translator's own dtype.
#
# On CUDA these make exactly the paper's calls (bf16 autocast, ``torch.cuda.synchronize``,
# the fast-inference executor); elsewhere no autocast and fast-inference level 0.

PAPER_LEVEL = 6  # the fast-inference level of the paper runs (CUDA only)


def autocast(device, dtype=torch.bfloat16):
    """``torch.autocast("cuda", dtype=dtype)`` for a CUDA ``device``; no autocast elsewhere."""

    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def synchronize(device) -> None:
    """``torch.cuda.synchronize()`` (the current device) on CUDA; MPS waits for its queue."""

    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def empty_cache(device) -> None:
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def resolve_level(level, device) -> int:
    """The fast-inference level for ``device``: ``None`` means the paper's level on CUDA
    and 0 (eager) elsewhere. Levels above 0 need CUDA (kernels, Triton)."""

    device = torch.device(device)
    if level is None:
        return PAPER_LEVEL if device.type == "cuda" else 0
    level = int(level)
    if level > 0 and device.type != "cuda":
        raise ValueError(f"fast-inference level {level} needs a CUDA device (got {device.type}); "
                         "use level=0 (the eager reference path) on cpu or mps")
    return level

