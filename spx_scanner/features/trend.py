"""
features/trend.py
-----------------
趋势相关特征:EMA、EMA 粘合度、VWAP(每日重置)、趋势方向。

所有阈值从 config/params.yaml 读取。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config


def compute_ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """计算多周期 EMA 并附加到 DataFrame。

    列名格式:ema_<period>,如 ema_9, ema_21, ema_50, ema_200。

    Args:
        df: 含 close 列的 OHLCV DataFrame。
        periods: EMA 周期列表,None 则从 config 读取。

    Returns:
        含新 EMA 列的 DataFrame(副本)。
    """
    cfg = load_config()
    if periods is None:
        periods = cfg["features"]["trend"]["ema_periods"]

    out = df.copy()
    for p in periods:
        out[f"ema_{p}"] = out["close"].ewm(span=p, adjust=False).mean()
    return out


def compute_ema_compression(df: pd.DataFrame) -> pd.DataFrame:
    """计算 EMA 粘合度指标。

    定义:std([ema_9, ema_21, ema_50]) / close,值落在过去 lookback bar 的
    compression_pctile 分位以下视为粘合。

    前提:df 中已有 ema_9, ema_21, ema_50 列。

    Args:
        df: 含 ema_9/21/50 列的 DataFrame。

    Returns:
        含 ema_spread, ema_compression(bool) 列的 DataFrame。
    """
    cfg = load_config()
    lookback = cfg["features"]["trend"]["compression_lookback"]
    pctile = cfg["features"]["trend"]["compression_pctile"]

    out = df.copy()
    ema_cols = ["ema_9", "ema_21", "ema_50"]
    for c in ema_cols:
        if c not in out.columns:
            raise ValueError(f"缺少列 {c},请先调用 compute_ema()")

    out["ema_spread"] = out[ema_cols].std(axis=1) / out["close"]
    threshold = out["ema_spread"].rolling(lookback, min_periods=lookback // 2).quantile(pctile)
    out["ema_compression"] = out["ema_spread"] < threshold
    return out


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """计算每日重置的 VWAP 及 ±1σ 带。

    VWAP = cumsum(typical_price × volume) / cumsum(volume),按交易日重置。
    σ 带 = VWAP ± std(typical_price 与 VWAP 之差) × 1。

    Args:
        df: 含 open/high/low/close/volume 的 DataFrame,index 为 tz-aware。

    Returns:
        含 vwap, vwap_upper_1s, vwap_lower_1s 列的 DataFrame。
    """
    out = df.copy()
    typical = (out["high"] + out["low"] + out["close"]) / 3
    pv = typical * out["volume"]

    dates = out.index.normalize()
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = out["volume"].groupby(dates).cumsum()
    vwap = cum_pv / cum_vol

    # 日内滚动标准差(典型价格与 vwap 偏差)
    dev = typical - vwap
    dev_sq = dev ** 2
    cum_dev_sq = dev_sq.groupby(dates).cumsum()
    cum_count = out.groupby(dates).cumcount() + 1
    vwap_std = np.sqrt(cum_dev_sq / cum_count)

    out["vwap"] = vwap
    out["vwap_upper_1s"] = vwap + vwap_std
    out["vwap_lower_1s"] = vwap - vwap_std
    return out


def compute_trend_direction(df: pd.DataFrame) -> pd.DataFrame:
    """计算趋势方向标签。

    规则:
    - ema_21 在过去 5 bar 的斜率为正且 close > vwap → +1(多头)
    - ema_21 斜率为负且 close < vwap → -1(空头)
    - 其余 → 0(中性)

    前提:df 中已有 ema_21, vwap 列。

    Args:
        df: 含 ema_21, vwap, close 列的 DataFrame。

    Returns:
        含 trend_direction 列的 DataFrame。
    """
    out = df.copy()
    ema_slope = out["ema_21"].diff(5)
    above_vwap = out["close"] > out["vwap"]

    conditions = [
        (ema_slope > 0) & above_vwap,
        (ema_slope < 0) & ~above_vwap,
    ]
    choices = [1, -1]
    out["trend_direction"] = np.select(conditions, choices, default=0)
    return out


def compute_all_trend(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算所有趋势特征。

    Args:
        df: 原始 OHLCV DataFrame。

    Returns:
        含所有趋势特征列的 DataFrame。
    """
    out = compute_ema(df)
    out = compute_ema_compression(out)
    out = compute_vwap(out)
    out = compute_trend_direction(out)
    return out
