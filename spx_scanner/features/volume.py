"""
features/volume.py
------------------
量能相关特征:RVOL(按 time-of-day 归一化)、量价一致性信号。

关键:成交量必须按 time-of-day 归一化,不能直接和隔夜或不同时段比。

所有阈值从 config/params.yaml 读取。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config


def compute_rvol(df: pd.DataFrame) -> pd.DataFrame:
    """计算 time-of-day 归一化的相对成交量(RVOL)。

    对每个时间槽(如 09:30、09:31…),取过去 lookback_days 个同时段 bar 的
    滚动平均量作为基准,再计算比值。

    Args:
        df: 含 volume 列的 DataFrame,index 为 tz-aware DatetimeIndex。

    Returns:
        含 vol_tod_mean, rvol 列的 DataFrame。
    """
    cfg = load_config()
    lookback_days = cfg["features"]["volume"]["rvol_lookback_days"]

    out = df.copy()
    # 提取时间槽(只用时分,忽略日期)
    times = out.index.time

    # 按时间槽分组计算滚动均量
    tod_mean = out["volume"].copy().astype(float)
    for t in pd.unique(times):
        mask = times == t
        idx = out.index[mask]
        # 使用 expanding(min_periods=1) 再截断为 lookback_days
        series = out.loc[idx, "volume"].astype(float)
        rolling_mean = series.rolling(lookback_days, min_periods=1).mean()
        tod_mean.loc[idx] = rolling_mean.values

    out["vol_tod_mean"] = tod_mean
    out["rvol"] = out["volume"] / out["vol_tod_mean"].replace(0, np.nan)
    return out


def compute_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """基于 RVOL 计算量能信号列。

    前提:df 中已有 rvol, open, close 列。

    Args:
        df: 含 rvol, open, close 列的 DataFrame。

    Returns:
        含 is_high_vol, volume_surge, bullish_vol, bearish_vol 列的 DataFrame。
    """
    cfg = load_config()
    high_thr = cfg["features"]["volume"]["rvol_high_threshold"]   # 1.5
    surge_thr = cfg["features"]["volume"]["rvol_surge_threshold"] # 2.0

    out = df.copy()
    is_bullish_bar = out["close"] > out["open"]
    is_bearish_bar = out["close"] < out["open"]

    out["is_high_vol"] = out["rvol"] >= high_thr
    out["volume_surge"] = out["rvol"] >= surge_thr
    out["bullish_vol"] = is_bullish_bar & out["is_high_vol"]
    out["bearish_vol"] = is_bearish_bar & out["is_high_vol"]
    return out


def compute_all_volume(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算所有量能特征。

    Args:
        df: 原始 OHLCV DataFrame。

    Returns:
        含所有量能特征列的 DataFrame。
    """
    out = compute_rvol(df)
    out = compute_volume_signals(out)
    return out
