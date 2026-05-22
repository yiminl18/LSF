"""Generate Hydra config overrides that MDocAgent's predict.py needs.

MDocAgent's Hydra config tree expects:
    config/dataset/<name>.yaml   (relative to upstream/MDocAgent/)
    config/model/<name>.yaml

We write our generated YAMLs into ``mdocagent/config_overrides/<group>/<name>.yaml``
and have the predict.py subprocess discover them via a Hydra ``hydra.searchpath``
override (see ``baseline.agentic_mdocagent._run_predict_subprocess``). We never
write into the submodule's working tree — that used to dirty ``git status``
and fight ``git submodule update``.

``CONFIG_OVERRIDES_ROOT`` is the directory ``hydra.searchpath`` should point at;
it contains ``dataset/`` and ``model/`` subdirs (Hydra groups).
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).parent
CONFIG_OVERRIDES_ROOT = _HERE / "config_overrides"
_OVERRIDES_DIR = CONFIG_OVERRIDES_ROOT / "dataset"
_MODEL_OVERRIDES_DIR = CONFIG_OVERRIDES_ROOT / "model"

# Legacy locations — if a previous version of this module wrote into the
# submodule, those stale files still take precedence over our searchpath copies
# because upstream's ``config_path`` is the primary search root. Purge them so
# the current generation always wins.
_LEGACY_UPSTREAM_DATASET_CFG_DIR = _HERE / "upstream" / "MDocAgent" / "config" / "dataset"
_LEGACY_UPSTREAM_MODEL_CFG_DIR = _HERE / "upstream" / "MDocAgent" / "config" / "model"

_LSF_DATASET_YAML = """\
defaults:
  - base
  - _self_

# document_path is intentionally inherited from base (`./data/lsf/documents`)
# but NOT used by this baseline. The LSF entry-point (agentic_mdocagent.py)
# calls baseline.mdocagent.adapter.prepare_inputs() with an explicit PDF path
# and writes the rendered PNGs to extract_dir, so upstream's BaseDataset
# never reads document_path. Anyone running upstream's predict.py directly on
# our `lsf` dataset config needs to override document_path themselves.
name: lsf
"""


def _purge_legacy_upstream_writes(config_name_prefix: str = "lsf_openai_") -> None:
    """Remove stale lsf*.yaml files inside the submodule, if a previous version
    of this module copied them there. Safe to call repeatedly.
    """
    legacy_dataset = _LEGACY_UPSTREAM_DATASET_CFG_DIR / "lsf.yaml"
    if legacy_dataset.exists():
        legacy_dataset.unlink()
    if _LEGACY_UPSTREAM_MODEL_CFG_DIR.exists():
        for path in _LEGACY_UPSTREAM_MODEL_CFG_DIR.glob(f"{config_name_prefix}*.yaml"):
            path.unlink()


def generate_lsf_dataset_config() -> Path:
    """Write ``dataset/lsf.yaml`` into the overrides dir. Idempotent."""
    _OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    target = _OVERRIDES_DIR / "lsf.yaml"
    target.write_text(_LSF_DATASET_YAML, encoding="utf-8")
    return target


_NOOP_MODEL_YAML = """\
defaults:
  - base
  - _self_

model: noop
module_name: baseline.mdocagent.noop_model
class_name: NoOpModel
max_new_tokens: 1
"""


def generate_noop_model_config() -> str:
    """Write ``model/noop.yaml`` into the overrides dir. Idempotent.

    image_agent is routed to this NoOpModel in the LSF integration to skip
    redundant vision LLM calls (see ``noop_model.py`` for the rationale).
    """
    _MODEL_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    target = _MODEL_OVERRIDES_DIR / "noop.yaml"
    target.write_text(_NOOP_MODEL_YAML, encoding="utf-8")
    return "noop"


def generate_lsf_openai_model_config(
    model: str,
    *,
    config_name: str = "lsf_openai",
) -> str:
    """Write a Hydra OpenAI-compatible model config and return its config_name.

    The generated YAML points Hydra at ``baseline.mdocagent.openai_model.MyOpenAI``
    so all MDocAgent agents share LSF's Azure/OpenAI client and cost capture.
    """
    # max_new_tokens=1536 overrides the upstream `model/base.yaml` default of
    # 256. On Azure GPT-5 reasoning deployments, max_completion_tokens INCLUDES
    # hidden reasoning tokens — a 256-token budget can be entirely consumed by
    # chain-of-thought, leaving `content=""`. 1536 gives the visible answer ~1k
    # headroom even after a ~500-token CoT burst.
    #
    # NOTE: do NOT emit ``api_key:`` here — MyOpenAI reads OPENAI_API_KEY from
    # the env directly, and any Hydra config dump on error (HYDRA_FULL_ERROR)
    # would otherwise leak the resolved key into subprocess logs.
    model_yaml = (
        "defaults:\n"
        "  - base\n"
        "  - _self_\n"
        "\n"
        f"model: {json.dumps(model)}\n"
        "module_name: baseline.mdocagent.openai_model\n"
        "class_name: MyOpenAI\n"
        "max_new_tokens: 1536\n"
    )
    _MODEL_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    target = _MODEL_OVERRIDES_DIR / f"{config_name}.yaml"
    target.write_text(model_yaml, encoding="utf-8")
    return config_name
