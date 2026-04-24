"""
patterns/liquidity_sweep.py
----------------------------
Liquidity Sweep(P1):扫前高/前低/ORB 高低后迅速反转。

Put 触发(扫前高后反转):
  1. 当前 bar high > 任一关键位(pdh/orb_high/prior_hour_high 等)
  2. 当前 bar close < 被扫除的关键位(失败穿越)
  3. rvol > 1.5
  4. 上影线显著:upper_wick / range > 0.6

Call 触发:镜像(扫前低后反转)

典型场景:开盘拉高扫前高做空、午后砸盘扫前低做多
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config
from spx_scanner.patterns.base import Pattern, Signal


class LiquiditySweep(Pattern):
    """Liquidity Sweep pattern。"""

    name = "liquidity_sweep"
    default_hold_min = 15

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        cfg = load_config()
        p = cfg["patterns"]["liquidity_sweep"]
        self.sweep_levels     = p["sweep_levels"]     # ["pdh","pdl","orb_high","orb_low","prior_hour_high","prior_hour_low"]
        self.wick_ratio       = p["wick_ratio"]       # 0.6
        self.rvol_threshold   = p["rvol_threshold"]   # 1.5
        self.default_hold_min = p["default_hold_min"]
        self.stop_buffer      = p["stop_buffer_pct"]  # 0.002

        # 分离高/低关键位名称
        self._high_levels = [l for l in self.sweep_levels if "high" in l or l == "pdh"]
        self._low_levels  = [l for l in self.sweep_levels if "low"  in l or l == "pdl"]

    _REQUIRED = ["rvol", "high", "low", "open", "close"]

    def detect(self, df: pd.DataFrame) -> list[Signal]:
        missing = [c for c in self._REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"缺少特征列: {missing}")

        signals: list[Signal] = []
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        opens  = df["open"].values

        for i in range(1, len(df)):
            row = df.iloc[i]
            ts  = df.index[i]

            if pd.isna(row["rvol"]) or row["rvol"] < self.rvol_threshold:
                continue

            bar_range = highs[i] - lows[i]
            if bar_range < 1e-6:
                continue

            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]

            # ── PUT: 扫前高后反转 ────────────────────────────────────────
            swept_high, level_name_h = self._find_swept_high(row, highs[i])
            if swept_high is not None and closes[i] < swept_high:
                if upper_wick / bar_range >= self.wick_ratio:
                    conf = self._score(row["rvol"], upper_wick / bar_range)
                    stop = swept_high * (1 + self.stop_buffer)
                    target = closes[i] - (stop - closes[i]) * 2
                    signals.append(Signal(
                        timestamp=ts, symbol=self.symbol, pattern=self.name,
                        direction="put", confidence=conf, entry_price=closes[i],
                        context={"swept_level": round(swept_high, 3),
                                 "level_name": level_name_h,
                                 "upper_wick_ratio": round(upper_wick / bar_range, 3),
                                 "rvol": round(float(row["rvol"]), 2)},
                        suggested_hold_min=self.default_hold_min,
                        stop_level=round(stop, 3),
                        target_level=round(target, 3),
                    ))

            # ── CALL: 扫前低后反转 ───────────────────────────────────────
            swept_low, level_name_l = self._find_swept_low(row, lows[i])
            if swept_low is not None and closes[i] > swept_low:
                if lower_wick / bar_range >= self.wick_ratio:
                    conf = self._score(row["rvol"], lower_wick / bar_range)
                    stop = swept_low * (1 - self.stop_buffer)
                    target = closes[i] + (closes[i] - stop) * 2
                    signals.append(Signal(
                        timestamp=ts, symbol=self.symbol, pattern=self.name,
                        direction="call", confidence=conf, entry_price=closes[i],
                        context={"swept_level": round(swept_low, 3),
                                 "level_name": level_name_l,
                                 "lower_wick_ratio": round(lower_wick / bar_range, 3),
                                 "rvol": round(float(row["rvol"]), 2)},
                        suggested_hold_min=self.default_hold_min,
                        stop_level=round(stop, 3),
                        target_level=round(target, 3),
                    ))

        return signals

    def _find_swept_high(self, row: pd.Series, bar_high: float):
        """找到被当前 bar 刺破的最高关键位。"""
        best_level, best_name = None, None
        for col in self._high_levels:
            val = row.get(col, None)
            if val is None or pd.isna(val):
                continue
            if bar_high > val:
                if best_level is None or val > best_level:
                    best_level, best_name = val, col
        return best_level, best_name

    def _find_swept_low(self, row: pd.Series, bar_low: float):
        """找到被当前 bar 刺穿的最低关键位。"""
        best_level, best_name = None, None
        for col in self._low_levels:
            val = row.get(col, None)
            if val is None or pd.isna(val):
                continue
            if bar_low < val:
                if best_level is None or val < best_level:
                    best_level, best_name = val, col
        return best_level, best_name

    def _score(self, rvol: float, wick_ratio: float) -> float:
        score = 0.5
        if rvol > 2.0: score += 0.15
        if rvol > 3.0: score += 0.10
        if wick_ratio > 0.70: score += 0.15
        if wick_ratio > 0.80: score += 0.10
        return float(np.clip(score, 0, 1))

    def explain(self, signal: Signal) -> str:
        ctx = signal.context
        wick_key = "upper_wick_ratio" if signal.direction == "put" else "lower_wick_ratio"
        return (
            f"[Liquidity Sweep → {signal.direction.upper()}]\n"
            f"  扫除关键位: {ctx.get('level_name')} @ {ctx.get('swept_level')}\n"
            f"  影线比: {ctx.get(wick_key)}  RVOL: {ctx.get('rvol')}\n"
            f"  入场: {signal.entry_price:.3f}  止损: {signal.stop_level:.3f}"
            f"  目标: {signal.target_level:.3f}  置信度: {signal.confidence:.2f}"
        )
