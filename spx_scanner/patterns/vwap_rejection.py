"""
patterns/vwap_rejection.py
---------------------------
VWAP Rejection(P1):价格从一侧触 VWAP 后被拒绝。

Put 触发(从下往上触 VWAP 被拒):
  1. 前 N=5 bar close 持续 < VWAP
  2. 当前 bar high >= VWAP 但 close < VWAP(插入但收回)
  3. 上影线显著:upper_wick / range > 0.5
  4. rvol > 1.2
  5. ema_21 斜率为负(趋势确认)

Call 触发:镜像
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config
from spx_scanner.patterns.base import Pattern, Signal


class VWAPRejection(Pattern):
    """VWAP Rejection pattern。"""

    name = "vwap_rejection"
    default_hold_min = 10

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        cfg = load_config()
        p = cfg["patterns"]["vwap_rejection"]
        self.prior_bars       = p["prior_bars_one_side"]     # 5
        self.wick_ratio       = p["upper_wick_ratio"]        # 0.5
        self.rvol_threshold   = p["rvol_threshold"]          # 1.2
        self.require_ema_align = p["require_ema21_aligned"]
        self.default_hold_min = p["default_hold_min"]
        self.stop_buffer      = p["stop_buffer_pct"]

    _REQUIRED = ["vwap", "rvol", "high", "low", "open", "close", "ema_21"]

    def detect(self, df: pd.DataFrame) -> list[Signal]:
        missing = [c for c in self._REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"缺少特征列: {missing}")

        signals: list[Signal] = []
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        opens  = df["open"].values
        vwap   = df["vwap"].values
        ema21  = df["ema_21"].values

        for i in range(self.prior_bars + 1, len(df)):
            row = df.iloc[i]
            ts  = df.index[i]

            if pd.isna(vwap[i]) or pd.isna(row["rvol"]):
                continue
            if row["rvol"] < self.rvol_threshold:
                continue

            bar_range = highs[i] - lows[i]
            if bar_range < 1e-6:
                continue

            prior_closes = closes[i - self.prior_bars : i]
            prior_vwap   = vwap[i - self.prior_bars : i]

            # ── PUT: 价格在 VWAP 下方,触 VWAP 被拒 ────────────────────
            below_vwap = prior_closes < prior_vwap
            if below_vwap.all():
                if highs[i] >= vwap[i] and closes[i] < vwap[i]:
                    upper_wick = highs[i] - max(opens[i], closes[i])
                    if upper_wick / bar_range >= self.wick_ratio:
                        # ema21 斜率确认
                        ema_slope_ok = True
                        if self.require_ema_align and i >= 5 and not pd.isna(ema21[i - 5]):
                            ema_slope_ok = ema21[i] < ema21[i - 5]
                        if ema_slope_ok:
                            conf = self._score(row["rvol"], upper_wick / bar_range,
                                               below_vwap.sum())
                            stop = vwap[i] * (1 + self.stop_buffer)
                            target = closes[i] - (stop - closes[i]) * 2
                            signals.append(Signal(
                                timestamp=ts, symbol=self.symbol, pattern=self.name,
                                direction="put", confidence=conf, entry_price=closes[i],
                                context={"vwap": round(vwap[i], 3),
                                         "upper_wick_ratio": round(upper_wick / bar_range, 3),
                                         "rvol": round(float(row["rvol"]), 2),
                                         "prior_below_bars": int(below_vwap.sum())},
                                suggested_hold_min=self.default_hold_min,
                                stop_level=round(stop, 3),
                                target_level=round(target, 3),
                            ))

            # ── CALL: 价格在 VWAP 上方,触 VWAP 被拒(下影线) ──────────
            above_vwap = prior_closes > prior_vwap
            if above_vwap.all():
                if lows[i] <= vwap[i] and closes[i] > vwap[i]:
                    lower_wick = min(opens[i], closes[i]) - lows[i]
                    if lower_wick / bar_range >= self.wick_ratio:
                        ema_slope_ok = True
                        if self.require_ema_align and i >= 5 and not pd.isna(ema21[i - 5]):
                            ema_slope_ok = ema21[i] > ema21[i - 5]
                        if ema_slope_ok:
                            conf = self._score(row["rvol"], lower_wick / bar_range,
                                               above_vwap.sum())
                            stop = vwap[i] * (1 - self.stop_buffer)
                            target = closes[i] + (closes[i] - stop) * 2
                            signals.append(Signal(
                                timestamp=ts, symbol=self.symbol, pattern=self.name,
                                direction="call", confidence=conf, entry_price=closes[i],
                                context={"vwap": round(vwap[i], 3),
                                         "lower_wick_ratio": round(lower_wick / bar_range, 3),
                                         "rvol": round(float(row["rvol"]), 2),
                                         "prior_above_bars": int(above_vwap.sum())},
                                suggested_hold_min=self.default_hold_min,
                                stop_level=round(stop, 3),
                                target_level=round(target, 3),
                            ))

        return signals

    def _score(self, rvol: float, wick_ratio: float, prior_bars: int) -> float:
        score = 0.5
        if rvol > 2.0:
            score += 0.15
        if wick_ratio > 0.65:
            score += 0.15
        if prior_bars >= self.prior_bars:
            score += 0.10
        return float(np.clip(score, 0, 1))

    def explain(self, signal: Signal) -> str:
        ctx = signal.context
        wick_key = "upper_wick_ratio" if signal.direction == "put" else "lower_wick_ratio"
        return (
            f"[VWAP Rejection → {signal.direction.upper()}]\n"
            f"  VWAP: {ctx.get('vwap')}  影线比: {ctx.get(wick_key)}\n"
            f"  RVOL: {ctx.get('rvol')}\n"
            f"  入场: {signal.entry_price:.3f}  止损: {signal.stop_level:.3f}"
            f"  目标: {signal.target_level:.3f}  置信度: {signal.confidence:.2f}"
        )
