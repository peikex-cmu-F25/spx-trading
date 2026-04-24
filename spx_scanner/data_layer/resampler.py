"""
data_layer/resampler.py
-----------------------
把 1min OHLCV 数据重采样为更粗粒度的 K 线。

规则:
- origin="09:30:00" 确保 bar 边界对齐开盘时间
- 使用 label="left" + closed="left"(bar 标签 = bar 开始时间)
- 过滤掉空 bar(RTH 外或无交易)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

RESAMPLE_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

# 09:30 = 9*60+30 = 570 分钟偏移,用于对齐 bar 边界到市场开盘
SESSION_OFFSET = "570min"


def resample_to_timeframe(
    df: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """把 1min DataFrame 重采样到指定时间框架。

    使用 origin="start_day" + offset=570min 将 bar 边界对齐到 09:30 开盘。

    Args:
        df: 1min OHLCV DataFrame,index 为 tz-aware DatetimeIndex。
        timeframe: pandas offset string,如 "3min"、"5min"、"15min"、"1h"。

    Returns:
        重采样后的 DataFrame,去除 NaN bar。
    """
    if df.empty:
        return df.copy()

    resampled = (
        df.resample(
            timeframe,
            origin="start_day",
            offset=SESSION_OFFSET,
            label="left",
            closed="left",
        )
        .agg(RESAMPLE_AGG)
        .dropna(subset=["open", "close"])  # 过滤无交易 bar
    )

    # 过滤 volume=0 的 bar(节假日 / 盘外)
    resampled = resampled[resampled["volume"] > 0]

    logger.debug(
        "重采样 1min(%d) → %s(%d)", len(df), timeframe, len(resampled)
    )
    return resampled


def resample_1m_to_3m(df: pd.DataFrame) -> pd.DataFrame:
    """1min → 3min K 线(主分析周期)。

    Args:
        df: 1min OHLCV DataFrame。

    Returns:
        3min OHLCV DataFrame。
    """
    return resample_to_timeframe(df, "3min")


def resample_1m_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """1min → 15min K 线(大结构过滤)。

    Args:
        df: 1min OHLCV DataFrame。

    Returns:
        15min OHLCV DataFrame。
    """
    return resample_to_timeframe(df, "15min")

