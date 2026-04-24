"""
backtest/metrics.py
-------------------
回测指标计算 — 基础 + 全套风险调整 + 置信度校准 + 学习曲线。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    """计算每个 (pattern, exit_strategy) 组合的回测指标。

    Args:
        trades: run_backtest() 返回的 trades DataFrame。

    Returns:
        metrics DataFrame,每行对应一个 (pattern, exit_strategy) 组合。
    """
    if trades.empty:
        return pd.DataFrame()

    records = []
    for (pattern, strategy), grp in trades.groupby(["pattern", "exit_strategy"]):
        rec = _calc_group(grp)
        rec["pattern"]       = pattern
        rec["exit_strategy"] = strategy
        records.append(rec)

    return pd.DataFrame(records).set_index(["pattern", "exit_strategy"])


def compute_metrics_flat(trades: pd.DataFrame) -> pd.DataFrame:
    """同 compute_metrics,但不要求 exit_strategy 列(用于任意分组后的子集)。

    Args:
        trades: 任意过滤后的 trades 子集。

    Returns:
        单行 DataFrame(或多行,按 pattern 分)。
    """
    if trades.empty:
        return pd.DataFrame()

    records = []
    for pattern, grp in trades.groupby("pattern"):
        rec = _calc_group(grp)
        rec["pattern"] = pattern
        records.append(rec)

    return pd.DataFrame(records).set_index("pattern")


def _calc_group(grp: pd.DataFrame) -> dict:
    pnl = grp["pnl_pct"].sort_index()  # 按时间排
    wins   = pnl > 0
    losses = pnl < 0

    n       = len(grp)
    win_rate = wins.mean()
    avg_win  = pnl[wins].mean() if wins.any() else 0.0
    avg_loss = pnl[losses].mean() if losses.any() else 0.0
    expect   = win_rate * avg_win + (1 - win_rate) * avg_loss

    total_win  = pnl[wins].sum()
    total_loss = pnl[losses].abs().sum()
    profit_factor = total_win / total_loss if total_loss > 0 else np.inf

    # Sharpe (trade-level)
    std_pnl = pnl.std(ddof=1)
    sharpe = (pnl.mean() / std_pnl * np.sqrt(n)) if std_pnl > 0 else 0.0

    # MFE / MAE
    avg_mfe = grp["mfe_pct"].mean() if "mfe_pct" in grp else 0.0
    avg_mae = grp["mae_pct"].mean() if "mae_pct" in grp else 0.0

    # 连胜/连败
    max_consec_wins, max_consec_losses = _max_consecutive(wins.values)

    # 最大回撤(基于累积 P&L)
    cumulative = pnl.cumsum()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_dd = float(drawdown.min())   # 负数

    # Calmar = 年化收益 / |max_drawdown|
    avg_pnl_per_trade = pnl.mean()
    trades_per_year = 252 * 43 / max(n, 1)  # 粗估:每年约 43 信号/pattern
    annual_return = avg_pnl_per_trade * trades_per_year
    calmar = annual_return / abs(max_dd) if max_dd < 0 else np.inf

    return {
        "n_trades":           n,
        "win_rate":           round(win_rate, 3),
        "avg_win_pct":        round(avg_win, 4),
        "avg_loss_pct":       round(avg_loss, 4),
        "expect_pct":         round(expect, 4),
        "profit_factor":      round(profit_factor, 3),
        "sharpe":             round(sharpe, 3),
        "avg_mfe_pct":        round(avg_mfe, 4),
        "avg_mae_pct":        round(avg_mae, 4),
        "total_pnl_pct":      round(float(pnl.sum()), 4),
        "max_consec_wins":    max_consec_wins,
        "max_consec_losses":  max_consec_losses,
        "max_drawdown_pct":   round(max_dd, 4),
        "calmar":             round(calmar, 3) if not np.isinf(calmar) else None,
    }


def _max_consecutive(is_win: np.ndarray) -> tuple[int, int]:
    """计算最大连胜和最大连败。"""
    max_w = max_l = cur_w = cur_l = 0
    for w in is_win:
        if w:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def confidence_calibration(
    trades: pd.DataFrame,
    n_bins: int = 5,
    min_bin_size: int = 3,
) -> pd.DataFrame:
    """计算置信度校准曲线数据。

    Args:
        trades: trades DataFrame,必须包含 confidence 和 pnl_pct 列。
        n_bins: 分桶数量。
        min_bin_size: 少于此数的 bin 会被标为统计不足。

    Returns:
        DataFrame with columns: bin_low, bin_high, n_trades, actual_win_rate, sufficient.
    """
    if trades.empty or "confidence" not in trades.columns:
        return pd.DataFrame()

    conf = trades["confidence"]
    wins = (trades["pnl_pct"] > 0).astype(int)

    edges = np.linspace(conf.min() - 1e-6, conf.max() + 1e-6, n_bins + 1)
    records = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        n = mask.sum()
        records.append({
            "bin_low":        round(lo, 3),
            "bin_high":       round(hi, 3),
            "n_trades":       int(n),
            "actual_win_rate": round(float(wins[mask].mean()), 3) if n > 0 else np.nan,
            "sufficient":     n >= min_bin_size,
        })

    return pd.DataFrame(records)


def learning_curve(
    trades: pd.DataFrame,
    n_blocks: int = 5,
) -> pd.DataFrame:
    """按时间分块计算胜率变化(检验稳定性)。

    Args:
        trades: 含 entry_time 和 pnl_pct 的 trades DataFrame。
        n_blocks: 分块数。

    Returns:
        DataFrame with columns: block_start, block_end, n_trades, win_rate, total_pnl_pct.
    """
    if trades.empty or "entry_time" not in trades.columns:
        return pd.DataFrame()

    df = trades.sort_values("entry_time").reset_index(drop=True)
    block_size = max(1, len(df) // n_blocks)
    records = []

    for i in range(0, len(df), block_size):
        block = df.iloc[i : i + block_size]
        if block.empty:
            continue
        records.append({
            "block_start":   block["entry_time"].iloc[0].date() if hasattr(block["entry_time"].iloc[0], "date") else block["entry_time"].iloc[0],
            "block_end":     block["entry_time"].iloc[-1].date() if hasattr(block["entry_time"].iloc[-1], "date") else block["entry_time"].iloc[-1],
            "n_trades":      len(block),
            "win_rate":      round(float((block["pnl_pct"] > 0).mean()), 3),
            "total_pnl_pct": round(float(block["pnl_pct"].sum()), 4),
        })

    return pd.DataFrame(records)
