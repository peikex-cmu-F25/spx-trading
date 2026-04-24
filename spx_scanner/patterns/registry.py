"""
patterns/registry.py
--------------------
注册并实例化所有 pattern。
"""

from __future__ import annotations

from spx_scanner.patterns.base import Pattern
from spx_scanner.patterns.failed_breakout import FailedBreakout
from spx_scanner.patterns.orb_breakout import ORBBreakout
from spx_scanner.patterns.squeeze_release import SqueezeRelease
from spx_scanner.patterns.vwap_rejection import VWAPRejection
from spx_scanner.patterns.liquidity_sweep import LiquiditySweep
from spx_scanner.patterns.last_hour_drift import LastHourDrift


def get_all_patterns(symbol: str = "SPY") -> list[Pattern]:
    """返回所有已启用 pattern 的实例列表。

    Args:
        symbol: 品种代码。

    Returns:
        Pattern 实例列表。
    """
    from spx_scanner.config_loader import load_config
    cfg = load_config()
    p = cfg["patterns"]

    patterns: list[Pattern] = []

    if p["failed_breakout"]["enabled"]:
        patterns.append(FailedBreakout(symbol=symbol))
    if p["orb_breakout"]["enabled"]:
        patterns.append(ORBBreakout(symbol=symbol))
    if p["squeeze_release"]["enabled"]:
        patterns.append(SqueezeRelease(symbol=symbol))
    if p["vwap_rejection"]["enabled"]:
        patterns.append(VWAPRejection(symbol=symbol))
    if p["liquidity_sweep"]["enabled"]:
        patterns.append(LiquiditySweep(symbol=symbol))
    if p["last_hour_drift"]["enabled"]:
        patterns.append(LastHourDrift(symbol=symbol))

    return patterns
