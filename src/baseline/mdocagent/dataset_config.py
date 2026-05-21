"""Generate Hydra config overrides that MDocAgent's predict.py needs.

MDocAgent's Hydra config tree expects:
    config/dataset/<name>.yaml   (relative to upstream/MDocAgent/)
    config/model/<name>.yaml

Upstream is a git submodule, so generated YAML files are written first to
``mdocagent/config_overrides/<group>/<name>.yaml`` (tracked source of truth)
and then copied into the submodule (untracked target consumed by Hydra) at
runtime. Idempotent: a copy that already matches is a no-op.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_HERE = Path(__file__).parent
_OVERRIDES_DIR = _HERE / "config_overrides" / "dataset"
_MODEL_OVERRIDES_DIR = _HERE / "config_overrides" / "model"
_UPSTREAM_DATASET_CFG_DIR = _HERE / "upstream" / "MDocAgent" / "config" / "dataset"
_UPSTREAM_MODEL_CFG_DIR = _HERE / "upstream" / "MDocAgent" / "config" / "model"

_LSF_DATASET_YAML = """\
defaults:
  - base
  - _self_

name: lsf
"""


def generate_lsf_dataset_config() -> Path:
    """Write ``dataset/lsf.yaml`` to the overrides dir and copy into upstream.

    Returns the path to the file inside ``upstream/MDocAgent/config/dataset/``.
    Idempotent.
    """
    _OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    source = _OVERRIDES_DIR / "lsf.yaml"
    source.write_text(_LSF_DATASET_YAML, encoding="utf-8")

    if _UPSTREAM_DATASET_CFG_DIR.exists():
        target = _UPSTREAM_DATASET_CFG_DIR / "lsf.yaml"
        if not target.exists() or target.read_text(encoding="utf-8") != _LSF_DATASET_YAML:
            shutil.copy2(source, target)
        return target
    return source


def generate_lsf_openai_model_config(
    model: str,
    *,
    config_name: str = "lsf_openai",
) -> str:
    """Write a Hydra OpenAI-compatible model config and return its config_name.

    The generated YAML points Hydra at ``baseline.mdocagent.openai_model.MyOpenAI``
    so all MDocAgent agents share LSF's Azure/OpenAI client and cost capture.
    """
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
    )
    _MODEL_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    source = _MODEL_OVERRIDES_DIR / f"{config_name}.yaml"
    source.write_text(model_yaml, encoding="utf-8")

    if _UPSTREAM_MODEL_CFG_DIR.exists():
        target = _UPSTREAM_MODEL_CFG_DIR / f"{config_name}.yaml"
        if not target.exists() or target.read_text(encoding="utf-8") != model_yaml:
            shutil.copy2(source, target)

    return config_name
