from __future__ import annotations
from typing import Any, Dict
import yaml

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg: Dict[str, Any], path: str = "config.yaml") -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

