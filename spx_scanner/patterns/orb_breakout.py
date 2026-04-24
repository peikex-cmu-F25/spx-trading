"""
patterns/orb_breakout.py
------------------------
ORB Breakout(P0 优先级):开盘前 30 分钟区间被突破。

Call 触发:
  1. minutes_from_open > 30(ORB 已形成,orb_high/orb_low 有值)
  2. close > orb_high(收在区间外)
  3. rvol > 1.3
  4. 突破 bar 为阳线(close > open)

Put 触发:镜像(close < orb_low)

过滤:
  - ORB 范围过小(< 过去 20 日均值 50%)→ skip
  - ORB 范围过大(> 过去 20 日均值 200%)→ 置信度降低
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config
from spx_scanner.patterns.base import Pattern, Signal


class ORBBreakout(Pattern):
    """Opening Range Breakout pattern。"""

    name = "orb_breakout"
    default_hold_min = 20

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        cfg = load_config()
        p = cfg["patterns"]["orb_breakout"]
        self.orb_duration_min  = p["orb_duration_min"]       # 30
        self.rvol_threshold    = p["rvol_threshold"]          # 1.3
        self.size_min_pctile   = p["orb_size_min_pctile"]    # 0.20
        self.size_max_pctile   = p["orb_size_max_pctile"]    # 0.95
        self.require_close_out = p["require_close_outside_range"]
        self.default_hold_min  = p["default_hold_min"]
        self.stop_at_midpoint  = p["stop_at_orb_midpoint"]

    _REQUIRED = ["orb_high", "orb_low", "rvol", "minutes_from_open"]

    def detect(self, df: pd.DataFrame) -> list[Signal]:
        missing = [c for c in self._REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"缺少特征列: {missing}")

        # 按日计算 ORB 范围历史分位
        orb_sizes = _compute_daily_orb_sizes(df)

        signals: list[Signal] = []
        closes = df["close"].values
        opens  = df["open"].values

        for i in range(1, len(df)):
            row = df.iloc[i]
            ts  = df.index[i]

            if pd.isna(row["orb_high"]) or pd.isna(row["orb_low"]):
                continue
            if row["minutes_from_open"] <= self.orb_duration_min:
                continue
            if pd.isna(row["rvol"]) or row["rvol"] < self.rvol_threshold:
                continue

            orb_h = row["orb_high"]
            orb_l = row["orb_low"]
            orb_size = orb_h - orb_l

            # 过滤异常 ORB 大小
            size_pctile = _orb_size_percentile(orb_sizes, ts, orb_size)
            if size_pctile is not None and size_pctile < self.size_min_pctile:
                continue

            conf_adj = 0.0
            if size_pctile is not None and size_pctile > self.size_max_pctile:
                conf_adj = -0.1  # 过大的 ORB 降低置信度

            # CALL:突破 orb_high
            if closes[i] > orb_h and (not self.require_close_out or closes[i] > orb_h):
                if closes[i] > opens[i]:  # 阳线
                    conf = float(np.clip(0.6 + conf_adj +
                                        (0.1 if row["rvol"] > 2.0 else 0.0) +
                                        (0.1 if row.get("trend_direction", 0) == 1 else 0.0),
                                        0, 1))
                    stop = orb_l if self.stop_at_midpoint else (orb_h + orb_l) / 2
                    target = closes[i] + (closes[i] - stop) * 1.5
                    signals.append(Signal(
                        timestamp=ts, symbol=self.symbol, pattern=self.name,
                        direction="call", confidence=conf, entry_price=closes[i],
                        context={"orb_high": round(orb_h, 3), "orb_low": round(orb_l, 3),
                                 "rvol": round(float(row["rvol"]), 2),
                                 "orb_size_pctile": round(size_pctile, 3) if size_pctile else -1},
                        suggested_hold_min=self.default_hold_min,
                        stop_level=round(stop, 3),
                        target_level=round(target, 3),
                    ))

            # PUT:跌破 orb_low
            elif closes[i] < orb_l:
                if closes[i] < opens[i]:  # 阴线
                    conf = float(np.clip(0.6 + conf_adj +
                                        (0.1 if row["rvol"] > 2.0 else 0.0) +
                                        (0.1 if row.get("trend_direction", 0) == -1 else 0.0),
                                        0, 1))
                    stop = orb_h if self.stop_at_midpoint else (orb_h + orb_l) / 2
                    target = closes[i] - (stop - closes[i]) * 1.5
                    signals.append(Signal(
                        timestamp=ts, symbol=self.symbol, pattern=self.name,
                        direction="put", confidence=conf, entry_price=closes[i],
                        context={"orb_high": round(orb_h, 3), "orb_low": round(orb_l, 3),
                                 "rvol": round(float(row["rvol"]), 2),
                                 "orb_size_pctile": round(size_pctile, 3) if size_pctile else -1},
                        suggested_hold_min=self.default_hold_min,
                        stop_level=round(stop, 3),
                        target_level=round(target, 3),
                    ))

        return signals

    def explain(self, signal: Signal) -> str:
        ctx = signal.context
        return (
            f"[ORB Breakout → {signal.direction.upper()}]\n"
            f"  ORB High: {ctx.get('orb_high')}  Low: {ctx.get('orb_low')}\n"
            f"  RVOL: {ctx.get('rvol')}  ORB大小分位: {ctx.get('orb_size_pctile')}\n"
            f"  入场: {signal.entry_price:.3f}  止损: {signal.stop_level:.3f}"
            f"  目标: {signal.target_level:.3f}  置信度: {signal.confidence:.2f}"
        )


def _compute_daily_orb_sizes(df: pd.DataFrame) -> dict:
    """按日期计算 ORB 范围,返回 {date: orb_size} dict。"""
    sizes = {}
    for date, day_df in df.groupby(df.index.normalize()):
        orb_h = day_df["orb_high"].dropna()
        orb_l = day_df["orb_low"].dropna()
        if len(orb_h) > 0 and len(orb_l) > 0:
            sizes[date.date()] = orb_h.iloc[0] - orb_l.iloc[0]
    return sizes


def _orb_size_percentile(orb_sizes: dict, ts: pd.Timestamp, current_size: float) -> float | None:
    """计算当前 ORB 范围在历史中的百分位。"""
    past = [v for d, v in orb_sizes.items() if d < ts.date()]
    if not past:
        return None
    past = sorted(past)
    rank = sum(1 for v in past if v <= current_size)
    return rank / len(past)
