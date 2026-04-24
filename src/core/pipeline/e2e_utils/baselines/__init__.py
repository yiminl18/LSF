"""RAG baseline plugin package.

Register new methods via the @register_baseline decorator,
and obtain instances via the get_baseline() factory.
Lazy imports: a missing dependency for one method does not affect others.
"""

import importlib

from core.pipeline.e2e_utils.baselines.base import BaseRAGBaseline, process_docs_baseline

_REGISTRY: dict[str, type[BaseRAGBaseline]] = {}

# Lazy-load mapping: config name -> module path
_LAZY_MODULES: dict[str, str] = {
    "rag-vanilla": "core.pipeline.e2e_utils.baselines.vanilla",
    "rag-raptor": "core.pipeline.e2e_utils.baselines.raptor",
    "rag-hippo": "core.pipeline.e2e_utils.baselines.hipporag",
    "rag-graph": "core.pipeline.e2e_utils.baselines.graphrag",
}


def register_baseline(name: str):
    """Decorator: register a baseline class into the global registry."""
    def decorator(cls: type[BaseRAGBaseline]) -> type[BaseRAGBaseline]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_loaded(config_name: str) -> None:
    """Import a baseline module on demand to trigger @register_baseline."""
    if config_name not in _REGISTRY and config_name in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[config_name])


def get_baseline(config_name: str, **kwargs) -> BaseRAGBaseline:
    """Factory: config name -> baseline instance."""
    _ensure_loaded(config_name)
    if config_name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys() | _LAZY_MODULES.keys()))
        raise ValueError(f"Unknown baseline: {config_name!r}. Available: {available}")
    return _REGISTRY[config_name](**kwargs)


def is_rag_baseline_config(config_name: str) -> bool:
    """Return True if config_name is a registered or available RAG baseline."""
    return config_name in _REGISTRY or config_name in _LAZY_MODULES


__all__ = [
    "BaseRAGBaseline",
    "get_baseline",
    "is_rag_baseline_config",
    "register_baseline",
    "process_docs_baseline",
]
