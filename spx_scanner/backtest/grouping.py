"""
backtest/grouping.py
--------------------
按环境维度对交易进行分组统计:
  - session_segment  (open / midday / close)
  - vix_regime       (low / mid / high)
  - dow              (Mon … Fri)
  - confidence_bucket(0.5-0.6 / 0.6-0.7 / 0.7-0.8 / 0.8+)
  - event flag       (is_opex / is_fomc)

典型用法:
    enriched = enrich_trades(trades, df)
    seg_tbl  = group_by_segment(enriched)
    heat     = session_heatmap(enriched)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.backtest.metrics import _calc_group


# ─────────────────────────────────────────────────────────────────────────────
# 数据富化
# ─────────────────────────────────────────────────────────────────────────────

_CONTEXT_COLS = [
    "session_segment", "vix_regime", "dow",
    "is_opex", "is_fomc", "is_cpi",
]


def enrich_trades(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """将 df 中的环境列按 entry_time 合并到 trades。

    Args:
        trades: run_backtest() 输出。
        df:     含所有特征的 3min DataFrame。

    Returns:
        enriched trades DataFrame。
    """
    if trades.empty:
        return trades.copy()

    available = [c for c in _CONTEXT_COLS if c in df.columns]
    if not available:
        return trades.copy()

    ctx = df[available].copy()
    ctx.index.name = "entry_time"
    ctx = ctx[~ctx.index.duplicated(keep="last")]

    enriched = trades.copy()
    for col in available:
        enriched[col] = enriched["entry_time"].map(ctx[col])

    # 规范化 dow: features 存 dayofweek 整数(0=Mon…4=Fri),转为英文日名
    _DOW_MAP = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday",
                5: "Saturday", 6: "Sunday"}
    if "dow" in enriched.columns:
        try:
            enriched["dow"] = enriched["dow"].map(
                lambda v: _DOW_MAP.get(int(v), str(v)) if pd.notna(v) else v
            )
        except (ValueError, TypeError):
            pass
    # 对 NaN(entry_time 没有精确匹配 df index 时)用 entry_time 直接推算
    nan_mask = enriched.get("dow", pd.Series(dtype=object)).isna()
    if nan_mask.any() or "dow" not in enriched.columns:
        enriched["dow"] = pd.to_datetime(enriched["entry_time"]).dt.day_name()

    # confidence_bucket
    enriched["confidence_bucket"] = pd.cut(
        enriched["confidence"],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 1.01],
        labels=["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8+"],
        right=False,
    )

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# 通用分组器
# ─────────────────────────────────────────────────────────────────────────────

def _group_by_col(
    trades: pd.DataFrame,
    col: str,
    extra_keys: list[str] | None = None,
) -> pd.DataFrame:
    """对 trades 按 col (+ extra_keys) 分组计算指标。

    Args:
        trades:     enriched trades DataFrame。
        col:        主分组列名。
        extra_keys: 附加分组键(如 ["pattern"])。

    Returns:
        长格式指标 DataFrame,含 col 值 + 所有 _calc_group 字段。
    """
    if trades.empty or col not in trades.columns:
        return pd.DataFrame()

    keys = ([col] + extra_keys) if extra_keys else [col]
    records = []
    for vals, grp in trades.groupby(keys, observed=True):
        if not isinstance(vals, tuple):
            vals = (vals,)
        rec = _calc_group(grp)
        for k, v in zip(keys, vals):
            rec[k] = v
        records.append(rec)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result = result.set_index(keys)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 各维度分组入口
# ─────────────────────────────────────────────────────────────────────────────

def group_by_segment(trades: pd.DataFrame, by_pattern: bool = False) -> pd.DataFrame:
    """按 session_segment 分组。

    Args:
        trades:     enriched trades。
        by_pattern: True 则同时按 pattern 分组。
    """
    extra = ["pattern"] if by_pattern else None
    return _group_by_col(trades, "session_segment", extra)


def group_by_vix(trades: pd.DataFrame, by_pattern: bool = False) -> pd.DataFrame:
    """按 vix_regime 分组。"""
    extra = ["pattern"] if by_pattern else None
    return _group_by_col(trades, "vix_regime", extra)


def group_by_dow(trades: pd.DataFrame, by_pattern: bool = False) -> pd.DataFrame:
    """按星期几分组。"""
    extra = ["pattern"] if by_pattern else None
    return _group_by_col(trades, "dow", extra)


def group_by_confidence(trades: pd.DataFrame, by_pattern: bool = False) -> pd.DataFrame:
    """按 confidence_bucket 分组。"""
    if trades.empty:
        return pd.DataFrame()
    if "confidence_bucket" not in trades.columns:
        trades = trades.copy()
        trades["confidence_bucket"] = pd.cut(
            trades["confidence"],
            bins=[0, 0.5, 0.6, 0.7, 0.8, 1.01],
            labels=["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8+"],
            right=False,
        )
    extra = ["pattern"] if by_pattern else None
    return _group_by_col(trades, "confidence_bucket", extra)


def group_by_event(
    trades: pd.DataFrame,
    event_col: str = "is_opex",
) -> pd.DataFrame:
    """按事件标志分组(is_opex / is_fomc 等)。

    Args:
        trades:    enriched trades。
        event_col: 要分组的布尔列名。
    """
    return _group_by_col(trades, event_col)


# ─────────────────────────────────────────────────────────────────────────────
# 热力图数据
# ─────────────────────────────────────────────────────────────────────────────

def session_heatmap(
    trades: pd.DataFrame,
    value: str = "win_rate",
) -> pd.DataFrame:
    """生成 pattern × session_segment 的胜率(或其他指标)矩阵。

    Args:
        trades: enriched trades。
        value:  要填入矩阵的指标名(win_rate / expect_pct / total_pnl_pct)。

    Returns:
        pivot table: rows=pattern, cols=session_segment。
    """
    if trades.empty or "session_segment" not in trades.columns:
        return pd.DataFrame()

    records = []
    for (pat, seg), grp in trades.groupby(["pattern", "session_segment"], observed=True):
        rec = _calc_group(grp)
        rec["pattern"]         = pat
        rec["session_segment"] = seg
        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="pattern", columns="session_segment", values=value)
    return pivot


def confidence_win_rate_table(trades: pd.DataFrame) -> pd.DataFrame:
    """confidence_bucket × pattern 的实际胜率矩阵。"""
    if trades.empty:
        return pd.DataFrame()

    if "confidence_bucket" not in trades.columns:
        trades = trades.copy()
        trades["confidence_bucket"] = pd.cut(
            trades["confidence"],
            bins=[0, 0.5, 0.6, 0.7, 0.8, 1.01],
            labels=["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8+"],
            right=False,
        )

    def _wr(x: pd.Series) -> float:
        return float((x > 0).mean())

    pivot = trades.pivot_table(
        index="confidence_bucket",
        columns="pattern",
        values="pnl_pct",
        aggfunc=_wr,
        observed=True,
    )
    return pivot
