"""
features/volatility.py
-----------------------
波动率相关特征:Bollinger Bands、BB 宽度分位、Squeeze、ATR、NR7。

所有阈值从 config/params.yaml 读取。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config


def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """计算布林带(Bollinger Bands)及 BB 宽度。

    bb_width = (bb_upper - bb_lower) / close。

    Args:
        df: 含 close 列的 DataFrame。

    Returns:
        含 bb_upper, bb_lower, bb_mid, bb_width 列的 DataFrame。
    """
    cfg = load_config()
    period = cfg["features"]["volatility"]["bb_period"]
    n_std = cfg["features"]["volatility"]["bb_std"]

    out = df.copy()
    rolling = out["close"].rolling(period, min_periods=period // 2)
    mid = rolling.mean()
    std = rolling.std(ddof=1)

    out["bb_mid"] = mid
    out["bb_upper"] = mid + n_std * std
    out["bb_lower"] = mid - n_std * std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["close"]
    return out


def compute_bb_width_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """计算 BB 宽度的历史百分位数。

    bb_width_pctile:过去 history_bars 个 bar 中,bb_width 的百分位排名。

    前提:df 中已有 bb_width 列。

    Args:
        df: 含 bb_width 列的 DataFrame。

    Returns:
        含 bb_width_pctile 列的 DataFrame。
    """
    cfg = load_config()
    history = cfg["features"]["volatility"]["bb_width_history_bars"]

    out = df.copy()
    # rank / count = 百分位
    out["bb_width_pctile"] = (
        out["bb_width"]
        .rolling(history, min_periods=20)
        .rank(pct=True)
    )
    return out


def compute_squeeze(df: pd.DataFrame) -> pd.DataFrame:
    """识别波动率压缩(Squeeze)状态。

    is_squeeze:bb_width_pctile < squeeze_pctile 且持续 ≥ squeeze_duration_bars。

    前提:df 中已有 bb_width_pctile 列。

    Args:
        df: 含 bb_width_pctile 列的 DataFrame。

    Returns:
        含 is_narrow(当前 bar 本身是否在压缩分位以下)、
        squeeze_duration(连续压缩 bar 数)、is_squeeze(bool) 列的 DataFrame。
    """
    cfg = load_config()
    pctile_max = cfg["features"]["volatility"]["squeeze_pctile"]
    min_duration = cfg["features"]["volatility"]["squeeze_duration_bars"]

    out = df.copy()
    is_narrow = out["bb_width_pctile"] < pctile_max

    # 连续计数
    group = (~is_narrow).cumsum()
    out["squeeze_duration"] = is_narrow.groupby(group).cumsum().where(is_narrow, 0).astype(int)
    out["is_squeeze"] = out["squeeze_duration"] >= min_duration
    return out


def compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    """计算 ATR(Average True Range)及与历史均值的比值。

    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    atr_ratio = atr_14 / atr_14.rolling(400).mean()

    Args:
        df: 含 high/low/close 列的 DataFrame。

    Returns:
        含 tr, atr_14, atr_ratio 列的 DataFrame。
    """
    cfg = load_config()
    period = cfg["features"]["momentum"]["rsi_period"]  # 14 复用
    atr_period = 14  # ATR 标准值
    history = cfg["features"]["volatility"]["bb_width_history_bars"]

    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["tr"] = tr
    out["atr_14"] = tr.ewm(span=atr_period, adjust=False).mean()
    long_avg = out["atr_14"].rolling(history, min_periods=20).mean()
    out["atr_ratio"] = out["atr_14"] / long_avg
    return out


def compute_nr7(df: pd.DataFrame) -> pd.DataFrame:
    """标记 NR7:当前 bar range 是过去 7 bar 中最小的。

    Args:
        df: 含 high/low 列的 DataFrame。

    Returns:
        含 bar_range, is_nr7 列的 DataFrame。
    """
    out = df.copy()
    out["bar_range"] = out["high"] - out["low"]
    min_7 = out["bar_range"].rolling(7, min_periods=7).min()
    out["is_nr7"] = out["bar_range"] <= min_7
    return out


def compute_all_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算所有波动率特征。

    Args:
        df: 原始 OHLCV DataFrame。

    Returns:
        含所有波动率特征列的 DataFrame。
    """
    out = compute_bollinger_bands(df)
    out = compute_bb_width_percentile(out)
    out = compute_squeeze(out)
    out = compute_atr(out)
    out = compute_nr7(out)
    return out
