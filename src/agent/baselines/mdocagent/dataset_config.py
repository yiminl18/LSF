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

import shutil
from pathlib import Path

_OVERRIDES_DIR = Path(__file__).parent / "config_overrides" / "dataset"
_UPSTREAM_DATASET_CFG_DIR = (
    Path(__file__).parent / "upstream" / "MDocAgent" / "config" / "dataset"
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
