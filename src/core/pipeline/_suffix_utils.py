from typing import Optional


def strip_required_suffix(stem: str, suffix: str) -> Optional[str]:
    """严格移除指定后缀；不匹配时返回 None。"""
    if not stem.endswith(suffix):
        return None
    return stem.removesuffix(suffix)
