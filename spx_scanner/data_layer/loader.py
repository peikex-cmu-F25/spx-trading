"""
data_layer/loader.py
--------------------
读取 CSV / Parquet 格式的 OHLCV 数据,统一 schema 和时区。
支持 yfinance 下载作为 demo/fallback 用途。

统一 schema:
    index:  pd.DatetimeIndex, tz="US/Eastern"
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

EASTERN = pytz.timezone("US/Eastern")
REQUIRED_COLS = {"open", "high", "low", "close", "volume"}
RTH_START = "09:30"
RTH_END = "16:00"


def load_data(
    path: str | Path,
    tz: str = "US/Eastern",
    rth_only: bool = True,
) -> pd.DataFrame:
    """从 CSV 或 Parquet 文件读取 OHLCV 数据,统一时区并规范列名。

    Args:
        path: 文件路径,支持 .csv / .parquet。
        tz: 目标时区,默认 US/Eastern。
        rth_only: 是否只保留 Regular Trading Hours (09:30–16:00)。

    Returns:
        DataFrame,index 为 tz-aware DatetimeIndex,列为 open/high/low/close/volume。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺少必要列或格式不支持。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"不支持的文件格式: {suffix} (仅支持 .csv / .parquet)")

    df = _normalize_schema(df, tz=tz)

    if rth_only:
        df = filter_rth(df)

    logger.info("加载 %d 条记录,时间范围: %s ~ %s", len(df), df.index.min(), df.index.max())
    return df


def download_spy_data(
    symbol: str = "SPY",
    days: int = 20,
    interval: str = "1m",
    save_path: str | Path | None = "data/spy_1min.parquet",
    rth_only: bool = True,
) -> pd.DataFrame:
    """用 yfinance 分批下载 OHLCV 数据(主要用于 demo/开发阶段)。

    yfinance 免费版每次最多拉 7 天 1min 数据,此函数自动分批拉取并合并。

    Args:
        symbol: 股票代码,默认 "SPY"。
        days: 下载最近多少个自然日,默认 20(约 14 个交易日)。
        interval: 时间粒度,如 "1m"、"5m"。
        save_path: 保存路径,None 则不保存。
        rth_only: 是否只保留 RTH。

    Returns:
        规范化后的 DataFrame。
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("请先安装 yfinance: pip install yfinance") from e

    import datetime

    BATCH_DAYS = 7  # yfinance 1min 数据每次最多 7 天

    end = datetime.datetime.now(tz=datetime.timezone.utc)
    chunks = []
    remaining = days

    while remaining > 0:
        batch = min(remaining, BATCH_DAYS)
        start = end - datetime.timedelta(days=batch)
        logger.info("下载 %s %s: %s ~ %s", symbol, interval,
                    start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        raw = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if not raw.empty:
            chunks.append(raw)
        end = start
        remaining -= batch

    if not chunks:
        raise ValueError(f"yfinance 返回空数据,请检查 symbol={symbol} 和网络连接")

    raw_all = pd.concat(chunks[::-1])  # 按时间正序合并
    df = _normalize_schema(raw_all, tz="US/Eastern")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if rth_only:
        df = filter_rth(df)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        logger.info("数据已保存至 %s", save_path)

    logger.info("下载完成: %d 条记录,时间范围: %s ~ %s", len(df), df.index.min(), df.index.max())
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """过滤只保留 Regular Trading Hours (09:30–16:00 ET)。

    Args:
        df: 带 tz-aware DatetimeIndex 的 DataFrame。

    Returns:
        过滤后的 DataFrame。
    """
    return df.between_time(RTH_START, RTH_END, inclusive="left")


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _normalize_schema(df: pd.DataFrame, tz: str = "US/Eastern") -> pd.DataFrame:
    """统一列名(小写)、index 时区、列类型。"""
    # yfinance 1.2+ 返回 MultiIndex(Price, Ticker),拍平取 Price 层
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower()
    else:
        df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]

    # 确保有必要列
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}。现有列: {list(df.columns)}")

    df = df[["open", "high", "low", "close", "volume"]].copy()

    # 时区处理
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)

    df.index.name = "timestamp"

    # 类型
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(int)

    # 去重、排序
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df
