"""
config_loader.py
----------------
读取 config/params.yaml,提供嵌套 dict 访问。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).parent.parent / "config" / "params.yaml"
_cache: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取并缓存 params.yaml。

    Args:
        path: yaml 路径,None 则使用默认 config/params.yaml。

    Returns:
        配置 dict。
    """
    global _cache
    if path is None and _cache is not None:
        return _cache
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is None:
        _cache = cfg
    return cfg


def get(key_path: str, path: str | Path | None = None) -> Any:
    """按点分路径获取配置值,如 'features.volatility.bb_period'。"""
    cfg = load_config(path)
    parts = key_path.split(".")
    node = cfg
    for p in parts:
        node = node[p]
    return node
