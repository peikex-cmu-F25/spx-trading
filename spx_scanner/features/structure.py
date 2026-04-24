"""
features/structure.py
---------------------
价格结构特征:支撑/阻力识别、ORB(开盘区间)、前日高低收。

所有阈值从 config/params.yaml 读取。

注:resistance/support 检测采用"触碰计数"逻辑:
    在过去 lookback_bars 个 bar 中,high(或 low)进入关键价位的
    tolerance 范围内即视为一次触碰。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config


# ---------------------------------------------------------------------------
# ORB
# ---------------------------------------------------------------------------

def compute_orb(df: pd.DataFrame) -> pd.DataFrame:
    """计算每日开盘区间(ORB)高低点。

    ORB:每日 RTH 前 orb_duration_bars 根 3min bar 的最高价和最低价。
    对应 09:30–10:00(10 根 × 3min = 30min)。

    列:orb_high, orb_low(按日向前填充,方便后续 bar 使用)。

    Args:
        df: 含 high/low 列的 3min OHLCV DataFrame。

    Returns:
        含 orb_high, orb_low 列的 DataFrame。
    """
    cfg = load_config()
    n_bars = cfg["features"]["structure"]["orb_duration_bars"]

    out = df.copy()
    out["orb_high"] = np.nan
    out["orb_low"] = np.nan

    for date, day_df in out.groupby(out.index.normalize()):
        if len(day_df) < n_bars:
            continue
        orb_slice = day_df.iloc[:n_bars]
        orb_h = orb_slice["high"].max()
        orb_l = orb_slice["low"].min()
        out.loc[day_df.index, "orb_high"] = orb_h
        out.loc[day_df.index, "orb_low"] = orb_l

    # ORB 结束前的 bar 置 NaN(尚未形成)
    for date, day_df in out.groupby(out.index.normalize()):
        orb_end_idx = min(n_bars, len(day_df))
        out.loc[day_df.index[:orb_end_idx], "orb_high"] = np.nan
        out.loc[day_df.index[:orb_end_idx], "orb_low"] = np.nan

    return out


# ---------------------------------------------------------------------------
# 前日高低收
# ---------------------------------------------------------------------------

def compute_pdh_pdl_pdc(df: pd.DataFrame) -> pd.DataFrame:
    """计算前一日 high / low / close(PDH/PDL/PDC)。

    Args:
        df: 含 high/low/close 列的 DataFrame。

    Returns:
        含 pdh, pdl, pdc 列的 DataFrame。
    """
    out = df.copy()

    # 每日的 H/L/C
    daily = out.groupby(out.index.normalize()).agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
    )
    daily.index = pd.to_datetime(daily.index)

    # shift 1 天
    daily_shifted = daily.shift(1, freq="D")

    # 按日期映射回分钟级 index
    date_idx = out.index.normalize()
    pdh = date_idx.map(daily_shifted["day_high"])
    pdl = date_idx.map(daily_shifted["day_low"])
    pdc = date_idx.map(daily_shifted["day_close"])

    out["pdh"] = pdh.values.astype(float)
    out["pdl"] = pdl.values.astype(float)
    out["pdc"] = pdc.values.astype(float)
    return out


# ---------------------------------------------------------------------------
# 支撑/阻力
# ---------------------------------------------------------------------------

def compute_resistance_support(df: pd.DataFrame) -> pd.DataFrame:
    """识别有效阻力位和支撑位,并统计每根 bar 前的触碰次数。

    阻力位:过去 lookback_bars 内的滚动最高 high,被触碰 ≥ min_touches 次。
    支撑位:镜像(滚动最低 low)。

    新增列:
    - resistance_level:   当前最近的有效阻力位(float)
    - support_level:      当前最近的有效支撑位(float)
    - touches_to_resistance: 该阻力位在过去 lookback_bars 内被触碰次数
    - touches_to_support:    对称

    算法:用滚动窗口的 rolling_max / rolling_min 作为候选关键位,
    再计算窗口内 high / low 在 tolerance 范围内出现次数。

    Args:
        df: 含 high/low/close 列的 DataFrame。

    Returns:
        含结构特征列的 DataFrame。
    """
    cfg = load_config()
    lookback = cfg["features"]["structure"]["resistance_lookback_bars"]
    min_touches = cfg["features"]["structure"]["min_touches"]
    tol = cfg["features"]["structure"]["touch_tolerance_pct"]

    out = df.copy()

    # 滚动候选价位
    roll_max = out["high"].rolling(lookback, min_periods=lookback // 2).max()
    roll_min = out["low"].rolling(lookback, min_periods=lookback // 2).min()

    res_level = np.full(len(out), np.nan)
    sup_level = np.full(len(out), np.nan)
    res_touches = np.zeros(len(out), dtype=int)
    sup_touches = np.zeros(len(out), dtype=int)

    highs = out["high"].values
    lows = out["low"].values

    for i in range(lookback, len(out)):
        start = max(0, i - lookback)
        window_highs = highs[start:i]
        window_lows = lows[start:i]
        candidate_res = roll_max.iloc[i]
        candidate_sup = roll_min.iloc[i]

        if pd.isna(candidate_res):
            continue

        # 触碰计数:high 进入 candidate_res 的 tolerance 范围
        tol_res = candidate_res * tol
        n_res = np.sum(np.abs(window_highs - candidate_res) <= tol_res)
        if n_res >= min_touches:
            res_level[i] = candidate_res
            res_touches[i] = n_res

        tol_sup = candidate_sup * tol
        n_sup = np.sum(np.abs(window_lows - candidate_sup) <= tol_sup)
        if n_sup >= min_touches:
            sup_level[i] = candidate_sup
            sup_touches[i] = n_sup

    out["resistance_level"] = res_level
    out["support_level"] = sup_level
    out["touches_to_resistance"] = res_touches
    out["touches_to_support"] = sup_touches
    return out


# ---------------------------------------------------------------------------
# 前小时高低(用于 LiquiditySweep)
# ---------------------------------------------------------------------------

def compute_prior_hour_levels(df: pd.DataFrame) -> pd.DataFrame:
    """计算前一小时的高低点(每 bar 实时更新)。

    在 3min 周期下,1 小时 = 20 根 bar。向前滚动 20 bar 的最高/最低。
    排除当前 bar 本身(shift(1) 开始)。

    Args:
        df: 含 high/low 列的 DataFrame。

    Returns:
        含 prior_hour_high, prior_hour_low 列的 DataFrame。
    """
    out = df.copy()
    # 向前看 20 bar(不含当前),取最高/最低
    out["prior_hour_high"] = out["high"].shift(1).rolling(20, min_periods=5).max()
    out["prior_hour_low"] = out["low"].shift(1).rolling(20, min_periods=5).min()
    return out


def compute_all_structure(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算所有结构特征。

    Args:
        df: 原始 OHLCV DataFrame(3min 周期)。

    Returns:
        含所有结构特征列的 DataFrame。
    """
    out = compute_orb(df)
    out = compute_pdh_pdl_pdc(out)
    out = compute_resistance_support(out)
    out = compute_prior_hour_levels(out)
    return out
