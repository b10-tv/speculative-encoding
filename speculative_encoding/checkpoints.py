"""Translator checkpoints: load any on-disk format, split a milestone per model, save a
translator, and download one pair's files from the Hugging Face repo (``resolve``)."""

from __future__ import annotations

import json
from pathlib import Path
import torch
from speculative_encoding.translator.model import SourceGeometry, TargetGeometry, UnifiedConfig
from speculative_encoding.translator.model import UnifiedTranslator, _key
from dataclasses import asdict
from dataclasses import dataclass, field


# ---- Checkpoint formats ------------------------------------------------------------------------
# Load the paper's translator checkpoints, and split them into per-model files.
#
# :func:`load_checkpoint` reads any of the three on-disk formats, detected from the
# folder's files:
#
# * **Milestone** (a training bundle, as published with the paper): ``meta.json`` with the
#   training ``config``, ``encoder.pt``, ``decoder.pt`` and one ``adapter-<model>.pt`` per
#   target. Every model in ``config["endpoints"]`` is both a source and a target.
# * **Split** (:func:`split_bundle`): ``shared.pt`` (encoder and decoder blocks),
#   ``source-<model>.pt`` (that source's encoder gains, input maps and state),
#   ``target-<model>.pt`` (that target's head) and ``manifest.json``. One pair needs
#   ``shared.pt`` plus two small files, and no Hugging Face access.
# * **Export** (``save_translator``): ``meta.json`` with ``format``,
#   ``config``, ``sources`` and ``targets``, plus one ``translator.pt`` with every model.
#
# Where the files come from: ``resolve`` below.
#
# State-dict keys by owner: ``encoder.blocks.*`` and ``decoder.blocks.*`` are shared;
# ``encoder.{gains,maps,states}.<source>`` belong to one source, ``heads.<target>.*`` to
# one target.

SOURCE_PARAMETERS = ("gains", "maps", "states")
SPLIT_FORMAT = "speculative-encoding-split/1"


def catalog_geometries(models) -> tuple[dict[str, SourceGeometry], dict[str, TargetGeometry]]:
    """Source and target geometries of catalog ``models`` from their pinned Hugging Face
    configs: streams = layers + 1; ``layer_types`` from the text config, else all sliding
    (when the config has a window) or all full attention."""

    from transformers import AutoConfig

    from speculative_encoding.models import CATALOG

    sources, targets = {}, {}
    for key in models:
        spec = CATALOG[key]
        config = AutoConfig.from_pretrained(spec.hf_id, revision=spec.revision)
        text = config.get_text_config(decoder=True)
        layers = int(text.num_hidden_layers)
        layer_types = getattr(text, "layer_types", None)
        if layer_types is None or len(layer_types) != layers:
            window = (getattr(text, "sliding_window", None)
                      or getattr(text, "attention_chunk_size", None))
            layer_types = ["sliding_attention" if window is not None else "full_attention"] * layers
        sources[key] = SourceGeometry(streams=layers + 1, width=int(text.hidden_size))
        targets[key] = TargetGeometry(layers=layers, width=int(text.hidden_size),
                                      layer_types=tuple(layer_types))
    return sources, targets


def _select(requested, available, what) -> list[str]:
    if requested is None:
        return list(available)
    requested = [requested] if isinstance(requested, str) else list(requested)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise KeyError(f"{what} {unknown} not in this checkpoint ({list(available)})")
    return [key for key in available if key in requested]  # checkpoint order


def _torch_load(path: Path, map_location):
    # mmap keeps the multi-GB files out of RAM: only the tensors a load keeps are paged in.
    mmap = torch.device(map_location).type == "cpu"
    return torch.load(path, map_location=map_location, mmap=mmap, weights_only=True)


def _empty_translator(sources, targets, config) -> UnifiedTranslator:
    """A translator on the meta device (no random init, no memory), for :func:`_fill`."""

    with torch.device("meta"):
        return UnifiedTranslator(targets, source_geometries=sources, config=config)


def _fill(translator: UnifiedTranslator, state: dict, *, dtype=None, device="cpu"
          ) -> UnifiedTranslator:
    """Strict load of ``state`` into a meta-device translator, keeping the loaded tensors
    (no copy)."""

    translator.load_state_dict(state, strict=True, assign=True)
    for module in translator.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None and buffer.is_meta:
                # The only non-persistent buffer: routed-expert counts, zeros at init.
                assert name == "load_counts", f"unexpected non-persistent buffer {name}"
                module._buffers[name] = torch.zeros(buffer.shape, dtype=buffer.dtype, device=device)
    if dtype is not None:
        translator.to(dtype)
    return translator


def _check_keys(state: dict, expected: set[str], what: str) -> None:
    if set(state) != expected:
        missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
        raise ValueError(f"{what}: missing {missing[:8]} unexpected {unexpected[:8]}")


def load_bundle(folder, *, sources=None, targets=None, dtype=None, map_location="cpu"
                ) -> UnifiedTranslator:
    """A ``UnifiedTranslator`` from a training milestone folder.

    ``sources`` / ``targets`` (model keys, default every endpoint) choose which per-model
    weights are kept; shared blocks are always loaded. Every file is checked strictly
    against all endpoints, dropped weights included. ``dtype`` casts after loading
    (``None`` keeps the stored float32)."""

    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    config = meta["config"]
    if config.get("adapter") != "unified" or (folder / "moe.pt").exists():
        raise ValueError(f"{folder}: only unified-adapter milestones without a separate MoE export")
    endpoints = list(config["endpoints"])
    source_keys = _select(sources, endpoints, "sources")
    target_keys = _select(targets, endpoints, "targets")
    source_geometries, _ = catalog_geometries(source_keys)
    _, target_geometries = catalog_geometries(target_keys)
    unified = UnifiedConfig(
        width=config["clean_width"], heads=config["clean_heads"], ffn=config["clean_ffn"],
        experts=config["clean_experts"], top_k=config["mm_top"],
        encoder_blocks=config["encoder_depth"], decoder_blocks=config["trunk_blocks"],
        window=config["cross_window"], decoder_streams=config.get("unified_decoder_streams"),
        target_blocks=int(config.get("unified_target_blocks", 0)),
        output_bias=bool(config.get("unified_output_bias", True)))
    translator = _empty_translator(source_geometries, target_geometries, unified)

    expected = set(translator.encoder.state_dict())
    expected |= {f"{part}.{_key(key)}" for part in SOURCE_PARAMETERS
                 for key in endpoints if key not in source_keys}
    encoder = _torch_load(folder / "encoder.pt", map_location)
    _check_keys(encoder, expected, "encoder.pt")
    decoder = _torch_load(folder / "decoder.pt", map_location)
    _check_keys(decoder, set(translator.decoder.state_dict()), "decoder.pt")

    kept = translator.encoder.state_dict()
    state = {f"encoder.{name}": value for name, value in encoder.items() if name in kept}
    state.update({f"decoder.{name}": value for name, value in decoder.items()})
    for key in target_keys:
        adapter = _torch_load(folder / f"adapter-{key}.pt", map_location)
        state.update({f"heads.{_key(key)}.{name}": value for name, value in adapter.items()})
    return _fill(translator, state, dtype=dtype, device=map_location)


def _owner(name: str) -> tuple[str, str | None]:
    """``("shared", None)``, ``("source", key)`` or ``("target", key)`` for a state key."""

    parts = name.split(".")
    if parts[0] == "heads":
        return "target", parts[1]
    if parts[0] == "encoder" and parts[1] in SOURCE_PARAMETERS:
        return "source", parts[2]
    if parts[0] in ("encoder", "decoder") and parts[1] == "blocks":
        return "shared", None
    raise ValueError(f"unknown translator key {name}")


def split_bundle(src_folder, out_folder, dtype=torch.bfloat16) -> dict:
    """Write a milestone as per-model files (see the module docstring); returns the
    manifest. ``dtype`` casts floating tensors (``None`` keeps float32). Reads the bundle
    through mmap, one file's worth of tensors in memory at a time."""

    src_folder = Path(src_folder)
    meta = json.loads((src_folder / "meta.json").read_text())
    return write_split(load_bundle(src_folder), out_folder, dtype=dtype,
                       extra={key: meta.get(key) for key in ("arm", "stage", "step")})


def write_split(translator: UnifiedTranslator, out_folder, dtype=torch.bfloat16,
                extra=None) -> dict:
    """Write any ``UnifiedTranslator`` as a split folder; returns the manifest."""

    out_folder = Path(out_folder)
    state = translator.state_dict()
    groups: dict[str, dict] = {}
    for name, value in state.items():
        kind, key = _owner(name)
        file = "shared.pt" if kind == "shared" else f"{kind}-{key}.pt"
        groups.setdefault(file, {})[name] = value
    out_folder.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for file, tensors in groups.items():
        compact = {}
        for name, value in tensors.items():
            if dtype is not None and value.is_floating_point() and value.dtype != dtype:
                value = value.to(dtype)
            elif value.untyped_storage().nbytes() != value.numel() * value.element_size():
                value = value.clone()  # torch.save writes the whole storage
            compact[name] = value
        torch.save(compact, out_folder / file)
        sizes[file] = (out_folder / file).stat().st_size
    manifest = {
        "format": SPLIT_FORMAT,
        "dtype": str(dtype or torch.float32).removeprefix("torch."),
        "config": translator.config.to_dict(),
        "sources": {key: {"streams": g.streams, "width": g.width}
                    for key, g in translator.sources.items()},
        "targets": {key: {"layers": g.layers, "width": g.width,
                          "layer_types": list(g.layer_types) if g.layer_types else None}
                    for key, g in translator.targets.items()},
        "files": {"shared": "shared.pt",
                  "sources": {key: f"source-{_key(key)}.pt" for key in translator.sources},
                  "targets": {key: f"target-{_key(key)}.pt" for key in translator.targets}},
        "bytes": sizes,
        "extra": dict(extra or {}),
    }
    (out_folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(folder, sources, targets, dtype=None, map_location="cpu") -> UnifiedTranslator:
    """A ``UnifiedTranslator`` for ``sources`` -> ``targets`` from a split folder, reading
    only ``shared.pt`` and those models' files. ``dtype=None`` keeps the stored dtype."""

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("format") != SPLIT_FORMAT:
        raise ValueError(f"{folder}: not a {SPLIT_FORMAT} folder")
    source_keys = _select(sources, manifest["sources"], "sources")
    target_keys = _select(targets, manifest["targets"], "targets")
    source_geometries = {key: SourceGeometry(**manifest["sources"][key]) for key in source_keys}
    target_geometries = {}
    for key in target_keys:
        value = manifest["targets"][key]
        layer_types = tuple(value["layer_types"]) if value.get("layer_types") else None
        target_geometries[key] = TargetGeometry(int(value["layers"]), int(value["width"]),
                                                layer_types)
    config = UnifiedConfig.from_dict(manifest["config"])
    translator = _empty_translator(source_geometries, target_geometries, config)
    files = manifest["files"]
    files = [files["shared"], *(files["sources"][k] for k in source_keys),
             *(files["targets"][k] for k in target_keys)]
    state = {}
    for file in files:
        state.update(_torch_load(folder / file, map_location))
    return _fill(translator, state, dtype=dtype, device=map_location)


def checkpoint_format(folder) -> str:
    """``"split"``, ``"milestone"`` or ``"export"`` for a checkpoint folder; otherwise a
    ``ValueError`` listing what the folder holds."""

    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder}: no such checkpoint folder")
    names = sorted(path.name for path in folder.iterdir())
    if "manifest.json" in names:
        return "split"
    if "meta.json" in names:
        if "translator.pt" in names:
            return "export"
        if "encoder.pt" in names and "decoder.pt" in names:
            return "milestone"
    shown = ", ".join(names[:20]) if names else "nothing"
    if len(names) > 20:
        shown += f", ... ({len(names)} entries)"
    raise ValueError(
        f"{folder} is not a translator checkpoint. Expected one of: a split folder "
        "(manifest.json, shared.pt, source-*.pt, target-*.pt), a training milestone "
        "(meta.json, encoder.pt, decoder.pt, adapter-<model>.pt) or an export (meta.json, "
        f"translator.pt). Found: {shown}")


def load_export(folder, *, sources=None, targets=None, dtype=None, map_location="cpu"
                ) -> UnifiedTranslator:
    """A ``UnifiedTranslator`` from an export (``meta.json`` + ``translator.pt``), keeping
    only ``sources`` / ``targets`` (default: every model)."""

    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    config = UnifiedConfig.from_dict(meta["config"])
    all_sources = {key: SourceGeometry(**value) for key, value in meta["sources"].items()}
    all_targets = {}
    for key, value in meta["targets"].items():
        layer_types = value.get("layer_types")
        all_targets[key] = TargetGeometry(int(value["layers"]), int(value["width"]),
                                          None if layer_types is None else tuple(layer_types))
    source_keys = _select(sources, all_sources, "sources")
    target_keys = _select(targets, all_targets, "targets")
    state = _torch_load(folder / "translator.pt", map_location)
    every = _empty_translator(all_sources, all_targets, config)  # strict, every model
    _check_keys(state, set(every.state_dict()), "translator.pt")
    translator = _empty_translator({key: all_sources[key] for key in source_keys},
                                   {key: all_targets[key] for key in target_keys}, config)
    state = {name: state[name] for name in translator.state_dict()}
    return _fill(translator, state, dtype=dtype, device=map_location)


def load_checkpoint(path, sources=None, targets=None, dtype=None, map_location="cpu"
                    ) -> UnifiedTranslator:
    """A ``UnifiedTranslator`` from a split, milestone or export folder.
    ``sources`` / ``targets``: model keys to keep (default every model in the checkpoint);
    ``dtype`` casts after loading (``None`` keeps the stored dtype)."""

    kind = checkpoint_format(path)
    if kind == "split":
        return load_split(path, sources, targets, dtype=dtype, map_location=map_location)
    load = load_bundle if kind == "milestone" else load_export
    return load(path, sources=sources, targets=targets, dtype=dtype, map_location=map_location)


# ---- Saving a translator -----------------------------------------------------------------------
# Native checkpoint I/O: ``translator.pt`` plus ``meta.json``.

def save_translator(directory, translator: UnifiedTranslator, *, extra=None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(translator.state_dict(), directory / "translator.pt")
    metadata = {
        "format": 1,
        "config": translator.config.to_dict(),
        "sources": {k: asdict(v) for k, v in translator.sources.items()},
        "targets": {k: asdict(v) for k, v in translator.targets.items()},
        "extra": dict(extra or {}),
    }
    text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    (directory / "meta.json").write_text(text)


# ---- Downloading one pair ----------------------------------------------------------------------
# The released checkpoint on the Hugging Face Hub, and downloading only what one source -> target
# pair needs.
#
# ``RampPublic/speculative-encoding`` (``REPO_ID``, public) holds:
#
#     translator/            a training milestone: meta.json, encoder.pt, decoder.pt and
#                            adapter-<model>.pt per model (all seven models, depth 8)
#     compressor-128x/       compressor-step-002400.pt + config.json
#
# A pair downloads ``meta.json``, ``encoder.pt``, ``decoder.pt`` and ``adapter-<target>.pt``
# (or, for a folder holding a split from ``split_bundle``, ``manifest.json``, ``shared.pt`` and
# the pair's ``source-*.pt`` / ``target-*.pt``). The folder goes straight to ``from_pretrained``:
#
#     from speculative_encoding import checkpoints
#     from speculative_encoding.api import SpeculativeEncoder
#
#     found = checkpoints.resolve("qwen3-14b", "llama31-8b", ratios=[128])
#     enc = SpeculativeEncoder.from_pretrained(found.checkpoint, "qwen3-14b", "llama31-8b",
#                                              compressors=found.compressors)
#
# Another repo or layout: ``resolve(..., repo_id=, translator="<folder>",
# compressors="<folder with {ratio}>")``.

REPO_ID = "RampPublic/speculative-encoding"
TRANSLATOR_FOLDER = "translator"
COMPRESSOR_FOLDER = "compressor-{ratio}x"
COMPRESSOR_FILE = "compressor-step-002400.pt"
MILESTONE_FILES = ("meta.json", "encoder.pt", "decoder.pt")


@dataclass
class Resolved:
    checkpoint: Path  # a folder for checkpoints.load_checkpoint (milestone or split)
    compressors: dict = field(default_factory=dict)  # {ratio: compressor .pt path}
    files: list = field(default_factory=list)  # the repo paths downloaded
    revision: str | None = None  # the repo commit the files come from


def compressor_folder(ratio, compressors: str = COMPRESSOR_FOLDER) -> str:
    return compressors.format(ratio=f"{float(ratio):g}")


def pair_files(files, source: str, target: str, *, translator: str = TRANSLATOR_FOLDER,
               compressors: str = COMPRESSOR_FOLDER, ratios=(),
               manifest: dict | None = None) -> list[str]:
    """The repo paths one pair needs, from the repo's file list ``files`` (no download).
    For a split folder, pass its ``manifest`` (``manifest.json``'s content) to get the
    per-model files; without it only the manifest is listed. Raises ``FileNotFoundError``
    naming what the repo lacks."""

    files = set(files)
    sources = [source] + ([target] if ratios and target != source else [])
    if f"{translator}/manifest.json" in files:
        needed = [f"{translator}/manifest.json"]
        if manifest is not None:
            listed = manifest["files"]
            missing = [key for key in sources if key not in listed["sources"]]
            missing += [target] if target not in listed["targets"] else []
            if missing:
                raise FileNotFoundError(f"{translator}: no per-model files for {missing}")
            needed += [f"{translator}/{listed['shared']}",
                       *(f"{translator}/{listed['sources'][key]}" for key in sources),
                       f"{translator}/{listed['targets'][target]}"]
    else:
        needed = [f"{translator}/{name}" for name in (*MILESTONE_FILES, f"adapter-{target}.pt")]
    for ratio in ratios:
        base = compressor_folder(ratio, compressors)
        needed += [f"{base}/{COMPRESSOR_FILE}", f"{base}/config.json"]
    missing = [path for path in needed if path not in files]
    if missing:
        available = sorted({path.rsplit("/", 1)[0] for path in files if "/" in path})
        raise FileNotFoundError(f"not in the checkpoint repo: {missing}. "
                                f"Folders there: {available}")
    return needed


def resolve(source: str, target: str, *, ratios=(), repo_id: str = REPO_ID,
            translator: str = TRANSLATOR_FOLDER, compressors: str = COMPRESSOR_FOLDER,
            revision: str | None = None, local_dir=None, token=None) -> Resolved:
    """Download (or find in the Hugging Face cache) only the files ``source -> target``
    needs, with one compressor per ``ratios`` entry; returns the local checkpoint folder
    and compressor paths. ``local_dir``: download there instead of the Hugging Face cache.
    Uses the cached Hugging Face login unless ``token`` is given."""

    import json

    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=token).model_info(repo_id, revision=revision)
    files = [sibling.rfilename for sibling in info.siblings]
    commit = info.sha

    def fetch(path):  # every file from one commit, so they share one snapshot folder
        return Path(hf_hub_download(repo_id, path, revision=commit, local_dir=local_dir,
                                    token=token))

    layout = dict(translator=translator, compressors=compressors, ratios=ratios)
    manifest = None
    if f"{translator}/manifest.json" in files:
        manifest = json.loads(fetch(f"{translator}/manifest.json").read_text())
    needed = pair_files(files, source, target, manifest=manifest, **layout)
    local = {path: fetch(path) for path in needed}
    found = {float(ratio): local[f"{compressor_folder(ratio, compressors)}/{COMPRESSOR_FILE}"]
             for ratio in ratios}
    return Resolved(checkpoint=local[needed[0]].parent, compressors=found, files=needed,
                    revision=commit)
