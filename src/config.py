"""
Configuration management.

Loads project settings from a YAML file and creates necessary
directories.

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department: Computer Science and Engineering, Uttara University
"""

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Centralized configuration manager (singleton pattern)."""

    _instance = None

    def __new__(cls, config_path: str = "configs/config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: str) -> None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)

        self._create_directories()

    def _create_directories(self) -> None:
        for path in self._config.get("paths", {}).values():
            Path(path).mkdir(parents=True, exist_ok=True)

    def get(self, *keys: str, default: Any = None) -> Any:
        value = self._config
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)
        return value

    @property
    def paths(self) -> Dict[str, str]:
        return self._config.get("paths", {})

    @property
    def seed(self) -> int:
        return self._config.get("project", {}).get("random_seed", 42)


config = Config()
