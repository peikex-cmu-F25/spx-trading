"""
patterns/failed_breakout.py
----------------------------
Failed Breakout / Failed Breakdown pattern(P0 优先级)。

Put 触发(Failed Breakout → 做空):
  1. 存在有效 resistance_level(过去 lookback_bars ≥ min_touches 次触碰)
  2. 最近 breakout_window_bars 内 high > resistance_level(曾尝试突破)
  3. 当前 bar close < resistance_level(回落到阻力下方)
  4. ema_compression == True 或 bb_width_pctile < squeeze_pctile(突破前压缩)
  5. 当前 bar 是阴线(close < open)且 rvol > rvol_threshold
  6. 当前 bar close < ema_21

Call 触发(Failed Breakdown → 做多):镜像逻辑。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config
from spx_scanner.patterns.base import Pattern, Signal


class FailedBreakout(Pattern):
    """Failed Breakout / Failed Breakdown pattern。"""

    name = "failed_breakout"
    default_hold_min = 15

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        cfg = load_config()
        p = cfg["patterns"]["failed_breakout"]
        self.lookback_bars        = p["lookback_bars"]
        self.min_touches          = p["min_touches"]
        self.touch_tol            = p["touch_tolerance_pct"]
        self.breakout_window      = p["breakout_window_bars"]
        self.rvol_threshold       = p["rvol_threshold"]
        self.squeeze_pctile       = p["squeeze_pctile"]
        self.default_hold_min     = p["default_hold_min"]
        self.stop_buffer          = p["stop_buffer_pct"]
        w = p["confidence_weights"]
        self.w_base           = w["base"]
        self.w_many_touches   = w["many_touches"]
        self.w_high_volume    = w["high_volume"]
        self.w_deep_squeeze   = w["deep_squeeze"]
        self.w_trend_confirm  = w["trend_confirm"]
        self.w_below_ema50    = w["below_ema50"]

    # ------------------------------------------------------------------ #
    # 必要特征列
    # ------------------------------------------------------------------ #
    _REQUIRED = [
        "resistance_level", "support_level",
        "touches_to_resistance", "touches_to_support",
        "ema_21", "ema_50", "bb_width_pctile",
        "ema_compression", "rvol", "consec_down", "consec_up",
    ]

    def detect(self, df: pd.DataFrame) -> list[Signal]:
        """扫描 DataFrame,返回所有 Failed Breakout/Breakdown 信号。

        Args:
            df: 含所有特征列的 3min DataFrame。

        Returns:
            Signal 列表。
        """
        missing = [c for c in self._REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"缺少特征列: {missing}")

        signals: list[Signal] = []
        highs   = df["high"].values
        lows    = df["low"].values
        closes  = df["close"].values
        opens   = df["open"].values

        min_start = self.lookback_bars + self.breakout_window + 1
        for i in range(min_start, len(df)):
            row = df.iloc[i]
            ts  = df.index[i]

            # ── PUT 信号(Failed Breakout → 做空) ─────────────────────
            sig_put = self._check_put(df, i, row, ts, highs, lows, closes, opens)
            if sig_put:
                signals.append(sig_put)

            # ── CALL 信号(Failed Breakdown → 做多) ───────────────────
            sig_call = self._check_call(df, i, row, ts, highs, lows, closes, opens)
            if sig_call:
                signals.append(sig_call)

        return signals

    def _resistance_from_history(self, highs: np.ndarray, i: int) -> tuple[float, int]:
        """从历史窗口(不含近期 breakout_window)计算阻力位和触碰次数。

        只看 [i-lookback_bars, i-breakout_window] 的 bar,让近期 bar 能"突破"它。
        """
        old_start = i - self.lookback_bars
        old_end   = i - self.breakout_window   # 不含近期 breakout_window 根
        if old_start < 0 or old_end <= old_start:
            return np.nan, 0

        old_highs = highs[old_start:old_end]
        if len(old_highs) == 0:
            return np.nan, 0

        level = old_highs.max()
        tol_abs = level * self.touch_tol
        touches = int(np.sum(np.abs(old_highs - level) <= tol_abs))
        return level, touches

    def _support_from_history(self, lows: np.ndarray, i: int) -> tuple[float, int]:
        """从历史窗口计算支撑位和触碰次数(阻力的镜像)。"""
        old_start = i - self.lookback_bars
        old_end   = i - self.breakout_window
        if old_start < 0 or old_end <= old_start:
            return np.nan, 0

        old_lows = lows[old_start:old_end]
        if len(old_lows) == 0:
            return np.nan, 0

        level = old_lows.min()
        tol_abs = level * self.touch_tol
        touches = int(np.sum(np.abs(old_lows - level) <= tol_abs))
        return level, touches

    def _check_put(self, df, i, row, ts, highs, lows, closes, opens) -> Signal | None:
        """检测 Failed Breakout(Put 方向)。

        阻力位取自不含近期 breakout_window 的历史,让近期 bar 能真实"突破"。
        """
        # 条件 1: 有效历史阻力位
        res_level, touches = self._resistance_from_history(highs, i)
        if np.isnan(res_level) or touches < self.min_touches:
            return None

        # 条件 2: 近期 breakout_window_bars 内高点曾突破阻力
        win_highs = highs[i - self.breakout_window : i]
        if not (win_highs > res_level).any():
            return None

        # 条件 3: 当前 bar 收在阻力下方
        if closes[i] >= res_level:
            return None

        # 条件 4: 突破前有压缩(EMA 粘合 或 BB 分位低)
        bb_pctile = row.get("bb_width_pctile", 1.0)
        ema_comp  = bool(row.get("ema_compression", False))
        if not (ema_comp or (not pd.isna(bb_pctile) and bb_pctile < self.squeeze_pctile)):
            return None

        # 条件 5: 阴线 + 放量
        if closes[i] >= opens[i]:     # 不是阴线
            return None
        if pd.isna(row["rvol"]) or row["rvol"] < self.rvol_threshold:
            return None

        # 条件 6: 跌破短期均线
        if closes[i] >= row["ema_21"]:
            return None

        # 置信度打分
        conf = self._score_put(row, touches, bb_pctile)

        # 止损:阻力位上方 stop_buffer
        stop = res_level * (1 + self.stop_buffer)
        target = closes[i] - (stop - closes[i]) * 2  # R:R = 1:2

        return Signal(
            timestamp=ts,
            symbol=self.symbol,
            pattern=self.name,
            direction="put",
            confidence=conf,
            entry_price=closes[i],
            context={
                "resistance_level":    round(res_level, 3),
                "touches":             int(touches),
                "rvol":                round(float(row["rvol"]), 2),
                "bb_width_pctile":     round(float(bb_pctile) if not pd.isna(bb_pctile) else -1, 3),
                "ema_compression":     ema_comp,
                "consec_down":         int(row.get("consec_down", 0)),
            },
            suggested_hold_min=self.default_hold_min,
            stop_level=round(stop, 3),
            target_level=round(target, 3),
        )

    def _check_call(self, df, i, row, ts, highs, lows, closes, opens) -> Signal | None:
        """检测 Failed Breakdown(Call 方向),与 Put 完全镜像。"""
        # 条件 1: 有效历史支撑位
        sup_level, touches = self._support_from_history(lows, i)
        if np.isnan(sup_level) or touches < self.min_touches:
            return None

        # 条件 2: 近期低点曾跌破支撑
        win_lows = lows[i - self.breakout_window : i]
        if not (win_lows < sup_level).any():
            return None

        # 条件 3: 当前 bar 收在支撑上方(反弹)
        if closes[i] <= sup_level:
            return None

        # 压缩条件
        bb_pctile = row.get("bb_width_pctile", 1.0)
        ema_comp  = bool(row.get("ema_compression", False))
        if not (ema_comp or (not pd.isna(bb_pctile) and bb_pctile < self.squeeze_pctile)):
            return None

        # 阳线 + 放量
        if closes[i] <= opens[i]:
            return None
        if pd.isna(row["rvol"]) or row["rvol"] < self.rvol_threshold:
            return None

        # 收在短期均线上方
        if closes[i] <= row["ema_21"]:
            return None

        conf = self._score_call(row, touches, bb_pctile)

        stop   = sup_level * (1 - self.stop_buffer)
        target = closes[i] + (closes[i] - stop) * 2

        return Signal(
            timestamp=ts,
            symbol=self.symbol,
            pattern=self.name,
            direction="call",
            confidence=conf,
            entry_price=closes[i],
            context={
                "support_level":    round(sup_level, 3),
                "touches":          int(touches),
                "rvol":             round(float(row["rvol"]), 2),
                "bb_width_pctile":  round(float(bb_pctile) if not pd.isna(bb_pctile) else -1, 3),
                "ema_compression":  ema_comp,
                "consec_up":        int(row.get("consec_up", 0)),
            },
            suggested_hold_min=self.default_hold_min,
            stop_level=round(stop, 3),
            target_level=round(target, 3),
        )

    def _score_put(self, row, touches: int, bb_pctile: float) -> float:
        score = self.w_base
        if touches >= 4:
            score += self.w_many_touches
        if not pd.isna(row["rvol"]) and row["rvol"] > 2.0:
            score += self.w_high_volume
        if not pd.isna(bb_pctile) and bb_pctile < 0.15:
            score += self.w_deep_squeeze
        if row.get("consec_down", 0) >= 2:
            score += self.w_trend_confirm
        if not pd.isna(row["ema_50"]) and row["close"] < row["ema_50"]:
            score += self.w_below_ema50
        return float(np.clip(score, 0.0, 1.0))

    def _score_call(self, row, touches: int, bb_pctile: float) -> float:
        score = self.w_base
        if touches >= 4:
            score += self.w_many_touches
        if not pd.isna(row["rvol"]) and row["rvol"] > 2.0:
            score += self.w_high_volume
        if not pd.isna(bb_pctile) and bb_pctile < 0.15:
            score += self.w_deep_squeeze
        if row.get("consec_up", 0) >= 2:
            score += self.w_trend_confirm
        if not pd.isna(row["ema_50"]) and row["close"] > row["ema_50"]:
            score += self.w_below_ema50
        return float(np.clip(score, 0.0, 1.0))

    def explain(self, signal: Signal) -> str:
        ctx = signal.context
        if signal.direction == "put":
            level_key, level_val = "resistance_level", ctx.get("resistance_level")
            return (
                f"[Failed Breakout → PUT]\n"
                f"  阻力位: {level_val}  触碰次数: {ctx.get('touches')}\n"
                f"  RVOL: {ctx.get('rvol')}  BB%分位: {ctx.get('bb_width_pctile')}\n"
                f"  EMA 粘合: {ctx.get('ema_compression')}  连续下跌: {ctx.get('consec_down')} bar\n"
                f"  入场: {signal.entry_price:.3f}  止损: {signal.stop_level:.3f}"
                f"  目标: {signal.target_level:.3f}  置信度: {signal.confidence:.2f}"
            )
        else:
            level_val = ctx.get("support_level")
            return (
                f"[Failed Breakdown → CALL]\n"
                f"  支撑位: {level_val}  触碰次数: {ctx.get('touches')}\n"
                f"  RVOL: {ctx.get('rvol')}  BB%分位: {ctx.get('bb_width_pctile')}\n"
                f"  EMA 粘合: {ctx.get('ema_compression')}  连续上涨: {ctx.get('consec_up')} bar\n"
                f"  入场: {signal.entry_price:.3f}  止损: {signal.stop_level:.3f}"
                f"  目标: {signal.target_level:.3f}  置信度: {signal.confidence:.2f}"
            )
