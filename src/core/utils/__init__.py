# -*- coding: utf-8 -*-
"""
Utility module.

Provides path management, parallel execution, text processing, and other shared utilities.
"""

from core.utils.paths import PathManager, PROJECT_ROOT
from core.utils.progress import create_progress, rich_tqdm, create_pipeline_progress
from core.utils.parallel import run_pool_with_progress
from core.utils.text import starts_with_number, starts_with_letter

__all__ = [
    "PathManager",
    "PROJECT_ROOT",
    "create_progress",
    "rich_tqdm",
    "create_pipeline_progress",
    "run_pool_with_progress",
    "starts_with_number",
    "starts_with_letter",
]
