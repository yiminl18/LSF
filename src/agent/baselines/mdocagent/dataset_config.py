"""Generate a Hydra dataset YAML for our LSF 10-K filings dataset.

MDocAgent's Hydra config system expects dataset configs at:
    config/dataset/<name>.yaml   (relative to upstream/MDocAgent/)

Because upstream/MDocAgent/ is a git submodule (DO NOT modify files inside it),
we write the generated YAML to a directory outside the submodule and either
symlink or copy it into place before running predict.py.

This module writes to:
    src/agent/baselines/mdocagent/config_overrides/dataset/lsf.yaml  (source)

And copies (not symlinks, to avoid submodule mutation perception) it to:
    upstream/MDocAgent/config/dataset/lsf.yaml  (target, created at runtime)

The generated YAML inherits from ``base`` and sets ``name: lsf`` so Hydra
resolves ``data_dir: ./data/lsf``, ``extract_path: ./tmp/lsf``, etc.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_OVERRIDES_DIR = Path(__file__).parent / "config_overrides" / "dataset"
_MODEL_OVERRIDES_DIR = Path(__file__).parent / "config_overrides" / "model"
_UPSTREAM_DATASET_CFG_DIR = (
    Path(__file__).parent / "upstream" / "MDocAgent" / "config" / "dataset"
)
_UPSTREAM_MODEL_CFG_DIR = (
    Path(__file__).parent / "upstream" / "MDocAgent" / "config" / "model"
)

_LSF_DATASET_YAML = """\
defaults:
  - base
  - _self_

name: lsf
"""


def generate_lsf_dataset_config() -> Path:
    """Write the lsf dataset YAML to config_overrides/ and copy to upstream.

    Returns the path to the file inside upstream/MDocAgent/config/dataset/.
    Idempotent: no-op if the target already exists and is identical.
    """
    _OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    source = _OVERRIDES_DIR / "lsf.yaml"
    source.write_text(_LSF_DATASET_YAML, encoding="utf-8")

    # Only copy into upstream if the submodule is present
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
    """Write a Hydra OpenAI-compatible model config for MDocAgent.

    MDocAgent composes ``model/<name>.yaml`` internally, so CLI overrides cannot
    change ``model/openai.yaml`` after composition. This generated config lets
    the wrapper route all MDocAgent agents to the requested runtime model without
    editing the upstream submodule source.
    """
    model_yaml = f"""\
defaults:
  - base
  - _self_

model: {json.dumps(model)}
api_key: ${{oc.env:OPENAI_API_KEY,}}
module_name: agent.baselines.mdocagent.openai_model
class_name: MyOpenAI
"""
    _MODEL_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    source = _MODEL_OVERRIDES_DIR / f"{config_name}.yaml"
    source.write_text(model_yaml, encoding="utf-8")

    if _UPSTREAM_MODEL_CFG_DIR.exists():
        target = _UPSTREAM_MODEL_CFG_DIR / f"{config_name}.yaml"
        if not target.exists() or target.read_text(encoding="utf-8") != model_yaml:
            shutil.copy2(source, target)

    return config_name
