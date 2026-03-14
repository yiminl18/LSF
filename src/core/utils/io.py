# -*- coding: utf-8 -*-
"""
I/O utility functions: stdout/stderr suppression, label file loading, etc.
"""

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import List


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Temporarily redirect stdout/stderr to devnull."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def load_labels(path: Path) -> List[dict]:
    """
    Load labels from a JSON file. Supports root as {"labels": [...]} or [...].
    """
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "labels" in data:
        return data["labels"]
    if isinstance(data, list):
        return data
    return []
