"""
patterns/last_hour_drift.py
----------------------------
Last Hour Drift(P2):尾盘(14:30 后)顺日内趋势加速。

触发条件:
  1. minutes_to_close <= 90(14:30 之后)
  2. 日内趋势明确:close vs vwap 和 ema_21 slope 方向一致
  3. 当前 bar 顺势(阳线顺多头 / 阴线顺空头)
  4. VWAP 未被反向穿越超过 N=3 bar

特殊:0DTE gamma pin 效应在这个时段最强。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config
from spx_scanner.patterns.base import Pattern, Signal


class LastHourDrift(Pattern):
    """Last Hour Drift pattern。"""

    name = "last_hour_drift"
    default_hold_min = 30

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        cfg = load_config()
        p = cfg["patterns"]["last_hour_drift"]
        self.trigger_after_min    = p["trigger_after_minute"]          # 300 (= 14:30)
        self.max_counter_bars     = p["max_counter_trend_bars"]        # 3
        self.require_vwap_align   = p["require_vwap_alignment"]
        self.default_hold_min     = p["default_hold_min"]
        self.force_exit_before    = p["force_exit_minutes_before_close"]  # 15

    _REQUIRED = ["minutes_from_open", "minutes_to_close", "vwap", "ema_21",
                 "close", "open", "rvol"]

    def detect(self, df: pd.DataFrame) -> list[Signal]:
        missing = [c for c in self._REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"缺少特征列: {missing}")

        signals: list[Signal] = []
        closes = df["close"].values
        opens  = df["open"].values
        vwap   = df["vwap"].values
        ema21  = df["ema_21"].values

        # 调整持仓时间:距收盘 <= force_exit_before 则不开新仓
        for i in range(10, len(df)):
            row = df.iloc[i]
            ts  = df.index[i]

            # 条件 1: 时段
            if row["minutes_from_open"] <= self.trigger_after_min:
                continue
            if row["minutes_to_close"] <= self.force_exit_before:
                continue

            if pd.isna(vwap[i]) or pd.isna(ema21[i]):
                continue

            # 条件 2: 日内趋势
            above_vwap = closes[i] > vwap[i]
            ema_slope  = ema21[i] - ema21[i - 5] if i >= 5 and not pd.isna(ema21[i-5]) else 0.0
            bullish_trend = above_vwap and ema_slope > 0
            bearish_trend = (not above_vwap) and ema_slope < 0

            if not (bullish_trend or bearish_trend):
                continue

            # 条件 3: 当前 bar 顺势
            is_bullish_bar = closes[i] > opens[i]
            is_bearish_bar = closes[i] < opens[i]

            if bullish_trend and not is_bullish_bar:
                continue
            if bearish_trend and not is_bearish_bar:
                continue

            # 条件 4: 最近 max_counter_bars 内 VWAP 未被反穿
            if self.require_vwap_align and i >= self.max_counter_bars:
                prior_closes = closes[i - self.max_counter_bars : i]
                prior_vwap   = vwap[i - self.max_counter_bars : i]
                if bullish_trend:
                    # 不允许近期有 close < vwap
                    if (prior_closes < prior_vwap).any():
                        continue
                else:
                    if (prior_closes > prior_vwap).any():
                        continue

            # 置信度
            rvol = row["rvol"] if not pd.isna(row["rvol"]) else 1.0
            hold = min(self.default_hold_min, int(row["minutes_to_close"]) - self.force_exit_before)
            hold = max(3, hold)

            if bullish_trend:
                conf = float(np.clip(0.5 + (0.1 if rvol > 1.3 else 0) +
                                     (0.1 if ema_slope > 0.05 else 0) +
                                     (0.1 if row["minutes_to_close"] <= 60 else 0), 0, 1))
                atr  = row.get("atr_14", 0) or 0
                stop = vwap[i] - atr * 0.5 if atr > 0 else vwap[i] * 0.999
                target = closes[i] + (closes[i] - stop) * 2
                signals.append(Signal(
                    timestamp=ts, symbol=self.symbol, pattern=self.name,
                    direction="call", confidence=conf, entry_price=closes[i],
                    context={"minutes_to_close": int(row["minutes_to_close"]),
                             "ema_slope": round(ema_slope, 4),
                             "rvol": round(float(rvol), 2),
                             "above_vwap": True},
                    suggested_hold_min=hold,
                    stop_level=round(stop, 3),
                    target_level=round(target, 3),
                ))
            else:
                conf = float(np.clip(0.5 + (0.1 if rvol > 1.3 else 0) +
                                     (0.1 if ema_slope < -0.05 else 0) +
                                     (0.1 if row["minutes_to_close"] <= 60 else 0), 0, 1))
                atr  = row.get("atr_14", 0) or 0
                stop = vwap[i] + atr * 0.5 if atr > 0 else vwap[i] * 1.001
                target = closes[i] - (stop - closes[i]) * 2
                signals.append(Signal(
                    timestamp=ts, symbol=self.symbol, pattern=self.name,
                    direction="put", confidence=conf, entry_price=closes[i],
                    context={"minutes_to_close": int(row["minutes_to_close"]),
                             "ema_slope": round(ema_slope, 4),
                             "rvol": round(float(rvol), 2),
                             "above_vwap": False},
                    suggested_hold_min=hold,
                    stop_level=round(stop, 3),
                    target_level=round(target, 3),
                ))

        return signals

    def explain(self, signal: Signal) -> str:
        ctx = signal.context
        return (
            f"[Last Hour Drift → {signal.direction.upper()}]\n"
            f"  距收盘: {ctx.get('minutes_to_close')}min  EMA21斜率: {ctx.get('ema_slope')}\n"
            f"  RVOL: {ctx.get('rvol')}  在VWAP上方: {ctx.get('above_vwap')}\n"
            f"  入场: {signal.entry_price:.3f}  止损: {signal.stop_level:.3f}"
            f"  目标: {signal.target_level:.3f}  置信度: {signal.confidence:.2f}"
        )
