"""
patterns/registry.py
--------------------
注册并实例化所有 pattern。
"""

from __future__ import annotations

from spx_scanner.patterns.base import Pattern
from spx_scanner.patterns.failed_breakout import FailedBreakout


def get_all_patterns(symbol: str = "SPY") -> list[Pattern]:
    """返回所有已启用 pattern 的实例列表。

    Args:
        symbol: 品种代码。

    Returns:
        Pattern 实例列表。
    """
    from spx_scanner.config_loader import load_config
    cfg = load_config()

    patterns: list[Pattern] = []

    if cfg["patterns"]["failed_breakout"]["enabled"]:
        patterns.append(FailedBreakout(symbol=symbol))

    return patterns
