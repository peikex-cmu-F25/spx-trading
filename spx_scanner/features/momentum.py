"""
features/momentum.py
--------------------
动量相关特征:RSI、MACD、连续同向 bar 计数。

所有阈值从 config/params.yaml 读取。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config


def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """计算标准 RSI(Wilder 平滑法)。

    RSI = 100 - 100 / (1 + RS),RS = avg_gain / avg_loss。
    使用 Wilder 平滑(EWM with alpha=1/period)与大多数平台对齐。

    Args:
        df: 含 close 列的 DataFrame。

    Returns:
        含 rsi_14 列的 DataFrame。
    """
    cfg = load_config()
    period = cfg["features"]["momentum"]["rsi_period"]

    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    return out


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD 指标及柱状图翻转信号。

    macd_line   = EMA(fast) - EMA(slow)
    macd_signal = EMA(macd_line, signal)
    macd_hist   = macd_line - macd_signal
    macd_hist_flip: 柱由负转正(+1)/ 正转负(-1)/ 无变化(0)

    Args:
        df: 含 close 列的 DataFrame。

    Returns:
        含 macd_line, macd_signal, macd_hist, macd_hist_flip 列的 DataFrame。
    """
    cfg = load_config()
    fast = cfg["features"]["momentum"]["macd_fast"]
    slow = cfg["features"]["momentum"]["macd_slow"]
    signal = cfg["features"]["momentum"]["macd_signal"]

    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()

    out["macd_line"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd_line"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]

    prev_hist = out["macd_hist"].shift(1)
    out["macd_hist_flip"] = np.where(
        (prev_hist < 0) & (out["macd_hist"] >= 0), 1,
        np.where((prev_hist >= 0) & (out["macd_hist"] < 0), -1, 0),
    ).astype(int)
    return out


def compute_consecutive_bars(df: pd.DataFrame) -> pd.DataFrame:
    """计算连续同向 bar 数。

    consec_up:  收盘价连续上涨的 bar 数(含当前 bar)
    consec_down: 收盘价连续下跌的 bar 数(含当前 bar)

    Args:
        df: 含 close 列的 DataFrame。

    Returns:
        含 consec_up, consec_down 列的 DataFrame。
    """
    out = df.copy()
    delta = out["close"].diff()

    up = (delta > 0).astype(int)
    dn = (delta < 0).astype(int)

    # 连续计数:每次方向改变时重置
    def _streak(s: pd.Series) -> pd.Series:
        group = (~s.astype(bool)).cumsum()
        return s.groupby(group).cumsum().where(s.astype(bool), 0)

    out["consec_up"] = _streak(up).astype(int)
    out["consec_down"] = _streak(dn).astype(int)
    return out


def compute_all_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算所有动量特征。

    Args:
        df: 原始 OHLCV DataFrame。

    Returns:
        含所有动量特征列的 DataFrame。
    """
    out = compute_rsi(df)
    out = compute_macd(out)
    out = compute_consecutive_bars(out)
    return out
