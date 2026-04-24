"""
backtest/metrics.py
-------------------
基础回测指标计算。
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


def _calc_group(grp: pd.DataFrame) -> dict:
    pnl = grp["pnl_pct"]
    wins = pnl > 0
    losses = pnl < 0

    n       = len(grp)
    win_rate = wins.mean()
    avg_win  = pnl[wins].mean() if wins.any() else 0.0
    avg_loss = pnl[losses].mean() if losses.any() else 0.0
    expect   = win_rate * avg_win + (1 - win_rate) * avg_loss

    total_win  = pnl[wins].sum()
    total_loss = pnl[losses].abs().sum()
    profit_factor = total_win / total_loss if total_loss > 0 else np.inf

    # Sharpe (trade-level, annualize by 252*130 bars per year)
    std_pnl = pnl.std(ddof=1)
    sharpe = (pnl.mean() / std_pnl * np.sqrt(252 * 130 / n)) if std_pnl > 0 else 0.0

    # MFE / MAE
    avg_mfe = grp["mfe_pct"].mean()
    avg_mae = grp["mae_pct"].mean()

    return {
        "n_trades":      n,
        "win_rate":      round(win_rate, 3),
        "avg_win_pct":   round(avg_win, 4),
        "avg_loss_pct":  round(avg_loss, 4),
        "expect_pct":    round(expect, 4),
        "profit_factor": round(profit_factor, 3),
        "sharpe":        round(sharpe, 3),
        "avg_mfe_pct":   round(avg_mfe, 4),
        "avg_mae_pct":   round(avg_mae, 4),
        "total_pnl_pct": round(pnl.sum(), 4),
    }
