"""
patterns/squeeze_release.py
----------------------------
Squeeze Release(P1):波动率极度压缩后单边爆发。

触发条件:
  1. is_squeeze == True 已持续 >= squeeze_duration_bars
  2. 当前 bar close > bb_upper(Call)或 close < bb_lower(Put)
  3. rvol > 1.5
  4. atr_ratio < 0.7(全局 ATR 也偏低,压缩是真实的)

方向:顺突破方向
置信度:压缩越深 + 量能越大分越高
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config
from spx_scanner.patterns.base import Pattern, Signal


class SqueezeRelease(Pattern):
    """Squeeze Release pattern。"""

    name = "squeeze_release"
    default_hold_min = 20

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        cfg = load_config()
        p = cfg["patterns"]["squeeze_release"]
        self.squeeze_duration = p["squeeze_duration_bars"]  # 5
        self.bb_pctile_max    = p["bb_width_pctile_max"]   # 0.20
        self.rvol_threshold   = p["rvol_threshold"]         # 1.5
        self.atr_ratio_max    = p["atr_ratio_max"]          # 0.7
        self.default_hold_min = p["default_hold_min"]
        self.stop_atr         = p["stop_buffer_atr"]        # 1.0

    _REQUIRED = ["is_squeeze", "squeeze_duration", "bb_upper", "bb_lower",
                 "rvol", "atr_14", "atr_ratio", "bb_width_pctile"]

    def detect(self, df: pd.DataFrame) -> list[Signal]:
        missing = [c for c in self._REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"缺少特征列: {missing}")

        signals: list[Signal] = []
        closes = df["close"].values

        for i in range(self.squeeze_duration + 1, len(df)):
            row = df.iloc[i]
            ts  = df.index[i]

            # 条件 1: is_squeeze 已持续足够久(squeeze_duration 列记录连续 bar 数)
            if not row["is_squeeze"]:
                continue
            if row["squeeze_duration"] < self.squeeze_duration:
                continue

            # 条件 3: 放量
            if pd.isna(row["rvol"]) or row["rvol"] < self.rvol_threshold:
                continue

            # 条件 4: 全局 ATR 也低
            if not pd.isna(row["atr_ratio"]) and row["atr_ratio"] >= self.atr_ratio_max:
                continue

            atr = row["atr_14"] if not pd.isna(row["atr_14"]) else 0.0
            bb_pctile = row.get("bb_width_pctile", 0.5)
            squeeze_dur = int(row["squeeze_duration"])

            # 条件 2: 突破 BB
            if closes[i] > row["bb_upper"]:
                conf = self._score(row["rvol"], bb_pctile, squeeze_dur)
                stop = closes[i] - self.stop_atr * atr if atr > 0 else row["bb_lower"]
                target = closes[i] + (closes[i] - stop) * 2
                signals.append(Signal(
                    timestamp=ts, symbol=self.symbol, pattern=self.name,
                    direction="call", confidence=conf, entry_price=closes[i],
                    context={"rvol": round(float(row["rvol"]), 2),
                             "bb_width_pctile": round(float(bb_pctile), 3),
                             "squeeze_duration": squeeze_dur,
                             "atr_ratio": round(float(row["atr_ratio"]) if not pd.isna(row["atr_ratio"]) else 0, 3)},
                    suggested_hold_min=self.default_hold_min,
                    stop_level=round(stop, 3),
                    target_level=round(target, 3),
                ))

            elif closes[i] < row["bb_lower"]:
                conf = self._score(row["rvol"], bb_pctile, squeeze_dur)
                stop = closes[i] + self.stop_atr * atr if atr > 0 else row["bb_upper"]
                target = closes[i] - (stop - closes[i]) * 2
                signals.append(Signal(
                    timestamp=ts, symbol=self.symbol, pattern=self.name,
                    direction="put", confidence=conf, entry_price=closes[i],
                    context={"rvol": round(float(row["rvol"]), 2),
                             "bb_width_pctile": round(float(bb_pctile), 3),
                             "squeeze_duration": squeeze_dur,
                             "atr_ratio": round(float(row["atr_ratio"]) if not pd.isna(row["atr_ratio"]) else 0, 3)},
                    suggested_hold_min=self.default_hold_min,
                    stop_level=round(stop, 3),
                    target_level=round(target, 3),
                ))

        return signals

    def _score(self, rvol: float, bb_pctile: float, squeeze_dur: int) -> float:
        score = 0.5
        if rvol > 2.0:
            score += 0.15
        if rvol > 3.0:
            score += 0.10
        if not pd.isna(bb_pctile) and bb_pctile < 0.10:
            score += 0.15
        elif not pd.isna(bb_pctile) and bb_pctile < 0.15:
            score += 0.10
        if squeeze_dur >= 10:
            score += 0.10
        return float(np.clip(score, 0, 1))

    def explain(self, signal: Signal) -> str:
        ctx = signal.context
        return (
            f"[Squeeze Release → {signal.direction.upper()}]\n"
            f"  压缩持续: {ctx.get('squeeze_duration')} bar  BB%分位: {ctx.get('bb_width_pctile')}\n"
            f"  RVOL: {ctx.get('rvol')}  ATR比: {ctx.get('atr_ratio')}\n"
            f"  入场: {signal.entry_price:.3f}  止损: {signal.stop_level:.3f}"
            f"  目标: {signal.target_level:.3f}  置信度: {signal.confidence:.2f}"
        )
