"""vLLM 0.24 on transformers 5.23: restore the two transformers symbols vLLM 0.24 still reads.

The translator needs transformers >= 5.21 (``Cache.get_recurrent_state``; the paper ran 5.23.0);
vLLM 0.24 was built against transformers <= 5.14. In between, transformers changed two things
vLLM reads:

* 5.15 folded Gemma 4's ``global_head_dim`` / ``num_global_key_value_heads`` into
  ``per_layer_config``, and the config became heterogeneous: reading ``config.head_dim`` or
  ``config.num_key_value_heads`` raises ``AmbiguousGlobalPerLayerAttributeError``. vLLM reads
  both; without the attributes it would build Gemma 4 12B's full-attention layers with head_dim
  256 and 8 K/V heads instead of 512 and 1. Fix: keep both attributes on the config (5.14's
  defaults: 512, None) and allow the global reads (5.14's values: the sliding layers'). Nothing
  in transformers 5.23 reads those attributes or relies on the error.
* 5.17 renamed ``PixtralRotaryEmbedding`` and removed ``position_ids_in_meshgrid``; vLLM's
  ``pixtral.py`` (imported by ``mistral3.py``, Ministral 3) imports both at module level. Fix:
  ``position_ids_in_meshgrid`` as in 5.14 and a placeholder ``PixtralRotaryEmbedding`` that only
  builds (vLLM builds the vision tower on the meta device, then drops it with
  ``language_model_only``).

vLLM imports model classes in a fresh interpreter, so the patch must load at interpreter start:
``python -m speculative_encoding.serving.transformers_compat --install`` copies this file into
site-packages with a ``.pth`` that calls ``install()``, which only registers an import hook
(stdlib only) and does nothing on transformers versions that still have the symbols.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys

_CONFIGS = {
    "transformers.models.gemma4.configuration_gemma4": ("Gemma4TextConfig",),
    "transformers.models.gemma4_unified.configuration_gemma4_unified": (
        "Gemma4UnifiedTextConfig",),
}
_PIXTRAL = "transformers.models.pixtral.modeling_pixtral"
PATCHED: list[str] = []


def _keep_global_head(module) -> None:
    for name in _CONFIGS[module.__name__]:
        cls = getattr(module, name, None)
        if cls is None or "global_head_dim" in getattr(cls, "__dataclass_fields__", {}):
            continue
        if "__post_init__" not in vars(cls) or getattr(cls.__post_init__, "_se_compat", False):
            continue
        original = cls.__post_init__

        def __post_init__(self, *args, _original=original, **kwargs):
            head = kwargs.get("global_head_dim", 512)
            heads = kwargs.get("num_global_key_value_heads")
            _original(self, *args, **kwargs)
            self.global_head_dim, self.num_global_key_value_heads = head, heads
            if getattr(self, "is_heterogeneous", False):
                self.allow_global_per_layer_attribute_access = True

        __post_init__._se_compat = True
        cls.__post_init__ = __post_init__
        PATCHED.append(f"{module.__name__}.{name}.global_head_dim")


def _pixtral(module) -> None:
    import torch
    from torch import nn

    if not hasattr(module, "position_ids_in_meshgrid"):
        def position_ids_in_meshgrid(patch_embeds_list, max_width):  # transformers 5.14.1
            positions = []
            for patch in patch_embeds_list:
                height, width = patch.shape[-2:]
                mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
                h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
                ids = h_grid * max_width + v_grid
                positions.append(ids[:, 0])
            return torch.cat(positions)

        module.position_ids_in_meshgrid = position_ids_in_meshgrid
        PATCHED.append(f"{_PIXTRAL}.position_ids_in_meshgrid")
    if not hasattr(module, "PixtralRotaryEmbedding"):
        class PixtralRotaryEmbedding(nn.Module):
            """Placeholder for vLLM 0.24's Pixtral vision tower: builds, never runs."""

            def __init__(self, config, device=None, layer_type=None):
                super().__init__()
                self.config = config

            def forward(self, *args, **kwargs):
                raise RuntimeError(
                    "transformers >= 5.17 has no PixtralRotaryEmbedding: serve Mistral 3 / "
                    "Pixtral checkpoints text-only (language_model_only=True) or use "
                    "transformers <= 5.16 for images")

        module.PixtralRotaryEmbedding = PixtralRotaryEmbedding
        PATCHED.append(f"{_PIXTRAL}.PixtralRotaryEmbedding")


_HOOKS = {**{name: _keep_global_head for name in _CONFIGS}, _PIXTRAL: _pixtral}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, hook):
        self.inner, self.hook = inner, hook

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.hook(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        hook = _HOOKS.get(fullname)
        if hook is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, hook)
        return spec


def install() -> None:
    """Patch the modules above when they load (now, for any already imported). Idempotent."""

    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    for name, hook in _HOOKS.items():
        if name in sys.modules:
            hook(sys.modules[name])


if __name__ == "__main__":
    if sys.argv[1:] == ["--install"]:
        import shutil
        import sysconfig
        from pathlib import Path

        site = Path(sysconfig.get_paths()["purelib"])
        shutil.copyfile(__file__, site / "se_transformers_compat.py")
        (site / "se_transformers_compat.pth").write_text(
            "import se_transformers_compat; se_transformers_compat.install()\n")
        print(f"installed {site / 'se_transformers_compat.pth'}")
    elif sys.argv[1:] == ["--check"]:  # in the target environment: import what vLLM imports
        install()
        import transformers
        from transformers.models.gemma4_unified.configuration_gemma4_unified import (
            Gemma4UnifiedTextConfig)
        from transformers.models.pixtral.modeling_pixtral import (  # noqa: F401
            PixtralRotaryEmbedding, position_ids_in_meshgrid)

        config = Gemma4UnifiedTextConfig(global_head_dim=512, num_global_key_value_heads=1)
        values = config.global_head_dim, config.num_global_key_value_heads, config.head_dim
        assert values == (512, 1, 256), vars(config)
        config.num_key_value_heads  # noqa: B018 (raises on an unpatched heterogeneous config)
        # The .pth copy patches first.
        patched = PATCHED + getattr(sys.modules.get("se_transformers_compat"), "PATCHED", [])
        print(f"transformers {transformers.__version__}: patched {patched or 'nothing'}")
    else:
        raise SystemExit(
            "usage: python -m speculative_encoding.serving.transformers_compat --install | --check")
