"""Configuration management system with variable substitution support."""

import re
import yaml
from pathlib import Path
from typing import Any, Dict
from copy import deepcopy


class Config:
    """Configuration manager with lazy loading and variable substitution."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.reload()
    
    def reload(self, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to configs/default.yaml relative to project root
            project_root = Path(__file__).resolve().parents[1]
            config_path = project_root / "configs" / "default.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Resolve variable substitutions
        self._config = self._resolve_variables(raw_config)
        self._project_root = Path(__file__).resolve().parents[1]
    
    def _resolve_variables(self, config: Dict, max_iter: int = 10) -> Dict:
        """Resolve ${...} variable references in config values."""
        config = deepcopy(config)
        pattern = re.compile(r'\$\{([^}]+)\}')
        
        for _ in range(max_iter):
            changed = False
            for key, value in self._iter_all_values(config):
                if isinstance(value, str) and '${' in value:
                    def replacer(match):
                        nonlocal changed
                        var_path = match.group(1)
                        resolved = self._get_nested(config, var_path)
                        if resolved is not None:
                            changed = True
                            return str(resolved)
                        return match.group(0)  # Keep unresolved
                    
                    new_value = pattern.sub(replacer, value)
                    self._set_nested(config, key, new_value)
            
            if not changed:
                break
        
        return config
    
    def _iter_all_values(self, d: Dict, prefix: str = ''):
        """Iterate all key paths and values in nested dict."""
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from self._iter_all_values(v, full_key)
            else:
                yield full_key, v
    
    def _get_nested(self, d: Dict, path: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split('.')
        current = d
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _set_nested(self, d: Dict, path: str, value: Any):
        """Set value in nested dict using dot notation."""
        keys = path.split('.')
        current = d
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'paths.pdf_dir')."""
        value = self._get_nested(self._config, path)
        return value if value is not None else default
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access."""
        return self.get(key)
    
    def __getattr__(self, name: str) -> Any:
        """Attribute-like access."""
        if name.startswith('_') or name in {'reload', 'get'}:
            return object.__getattribute__(self, name)
        return self._config.get(name)
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root
    
    def to_path(self, path_key: str) -> Path:
        """Convert config path to absolute Path object."""
        path_str = self.get(path_key)
        if path_str is None:
            raise ValueError(f"Path not found in config: {path_key}")
        
        path = Path(path_str)
        if not path.is_absolute():
            path = self.project_root / path
        return path


# Global config instance
_CONFIG = Config()


def get_config() -> Config:
    """Get the global config instance."""
    return _CONFIG


def reload_config(config_path: str = None):
    """Reload configuration from file."""
    _CONFIG.reload(config_path)


