"""
features/regime.py
------------------
市场环境特征:VIX 分位、时段标签、星期几、事件日(FOMC/CPI/OPEX)。

所有阈值从 config/params.yaml 读取。
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config

SESSION_START = datetime.time(9, 30)
SESSION_END = datetime.time(16, 0)


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算时段相关特征。

    新增列:
    - minutes_from_open:  距 09:30 的分钟数
    - minutes_to_close:   距 16:00 的分钟数
    - session_segment:    "open" / "midday" / "close"
    - dow:                0=周一 … 4=周五

    Args:
        df: 含 tz-aware DatetimeIndex 的 DataFrame。

    Returns:
        含时段特征列的 DataFrame。
    """
    cfg = load_config()
    segs = cfg["features"]["regime"]["session_segments"]

    out = df.copy()
    # 以分钟计算
    open_ts = pd.Timestamp("1970-01-01 09:30:00")
    close_ts = pd.Timestamp("1970-01-01 16:00:00")

    bar_time = out.index.to_series().apply(
        lambda ts: pd.Timestamp(f"1970-01-01 {ts.strftime('%H:%M:%S')}")
    )
    out["minutes_from_open"] = (bar_time - open_ts).dt.total_seconds() / 60
    out["minutes_to_close"] = (close_ts - bar_time).dt.total_seconds() / 60
    out["dow"] = out.index.dayofweek

    # session_segment
    open_end = _parse_time(segs["open"][1])
    close_start = _parse_time(segs["close"][0])

    bar_time_t = pd.to_datetime(out.index).time if not hasattr(out.index, "time") else out.index.time

    def _segment(t: datetime.time) -> str:
        if t < open_end:
            return "open"
        elif t >= close_start:
            return "close"
        else:
            return "midday"

    out["session_segment"] = [_segment(t) for t in bar_time_t]
    return out


def compute_vix_regime(
    df: pd.DataFrame,
    vix_series: pd.Series | None = None,
) -> pd.DataFrame:
    """在 DataFrame 上附加 VIX 分位标签。

    如果提供 vix_series(日线 VIX),按日期 merge;
    否则用 vix_level 列(如果已存在)。

    vix_regime:
    - "low"  : vix < 15
    - "mid"  : 15 ≤ vix < 20
    - "high" : vix ≥ 20

    Args:
        df: 含 tz-aware DatetimeIndex 的 DataFrame。
        vix_series: 可选,日线 VIX Series,index 为日期。

    Returns:
        含 vix_level, vix_regime 列的 DataFrame。
    """
    cfg = load_config()
    vix_low = cfg["features"]["regime"]["vix_low"]
    vix_high = cfg["features"]["regime"]["vix_high"]

    out = df.copy()

    if vix_series is not None:
        # 统一转为 datetime.date 再 map,避免 tz-aware vs tz-naive 不匹配
        vix_map = {pd.Timestamp(k).date(): v for k, v in vix_series.items()}
        date_arr = [ts.date() for ts in out.index]
        out["vix_level"] = [vix_map.get(d, np.nan) for d in date_arr]
    elif "vix_level" not in out.columns:
        out["vix_level"] = np.nan

    def _regime(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        if v < vix_low:
            return "low"
        if v < vix_high:
            return "mid"
        return "high"

    out["vix_regime"] = out["vix_level"].map(_regime)
    return out


def compute_event_flags(
    df: pd.DataFrame,
    events_path: str | Path | None = None,
) -> pd.DataFrame:
    """附加事件日标志:is_opex, is_fomc, is_cpi。

    从 data/events.csv 读取(columns: date, event_type, notes)。
    如果文件不存在则全部标为 False。

    Args:
        df: 含 tz-aware DatetimeIndex 的 DataFrame。
        events_path: events.csv 路径,None 则从 config 读取。

    Returns:
        含 is_opex, is_fomc, is_cpi 列的 DataFrame。
    """
    cfg = load_config()
    if events_path is None:
        events_path = Path(cfg["data"]["events_file"])

    out = df.copy()
    date_idx = pd.to_datetime(out.index.date)

    # 默认全 False
    out["is_opex"] = False
    out["is_fomc"] = False
    out["is_cpi"] = False

    if Path(events_path).exists():
        events = pd.read_csv(events_path, parse_dates=["date"])
        for _, row in events.iterrows():
            mask = date_idx == row["date"]
            etype = str(row["event_type"]).upper()
            if "OPEX" in etype:
                out.loc[out.index[mask.values], "is_opex"] = True
            if "FOMC" in etype:
                out.loc[out.index[mask.values], "is_fomc"] = True
            if "CPI" in etype:
                out.loc[out.index[mask.values], "is_cpi"] = True

    # OPEX 兜底:月度第三个周五
    out["is_opex"] = out["is_opex"] | _is_third_friday(out.index)
    return out


def compute_all_regime(
    df: pd.DataFrame,
    vix_series: pd.Series | None = None,
    events_path: str | Path | None = None,
) -> pd.DataFrame:
    """一次性计算所有 regime 特征。

    Args:
        df: 原始 OHLCV DataFrame。
        vix_series: 可选 VIX 日线 Series。
        events_path: 可选 events.csv 路径。

    Returns:
        含所有 regime 特征列的 DataFrame。
    """
    out = compute_session_features(df)
    out = compute_vix_regime(out, vix_series=vix_series)
    out = compute_event_flags(out, events_path=events_path)
    return out


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _parse_time(s: str) -> datetime.time:
    """解析 "HH:MM" 字符串为 datetime.time。"""
    h, m = s.split(":")
    return datetime.time(int(h), int(m))


def _is_third_friday(index: pd.DatetimeIndex) -> pd.Series:
    """判断日期是否为当月第三个周五(月度 OPEX)。"""
    dates = pd.Series(index.date, index=index)
    result = dates.apply(lambda d: _third_friday_of_month(d.year, d.month) == d)
    return result


def _third_friday_of_month(year: int, month: int) -> datetime.date:
    """返回指定年月的第三个周五。"""
    first_day = datetime.date(year, month, 1)
    # 第一个周五
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + datetime.timedelta(days=days_until_friday)
    return first_friday + datetime.timedelta(weeks=2)
