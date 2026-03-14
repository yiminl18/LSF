from typing import Optional


def strip_required_suffix(stem: str, suffix: str) -> Optional[str]:
    """Strictly remove the specified suffix; return None when it does not match."""
    if not stem.endswith(suffix):
        return None
    return stem.removesuffix(suffix)
