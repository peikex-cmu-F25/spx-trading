"""
viz/chart.py
------------
单信号复盘图:K线 + 均线 + VWAP + 入场/出场标记 + 量能。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spx_scanner.patterns.base import Signal


def plot_signal_context(
    df: pd.DataFrame,
    signal: "Signal",
    bars_before: int = 30,
    bars_after: int = 30,
    trade_result: dict | None = None,
) -> plt.Figure:
    """绘制单信号复盘图。

    Args:
        df: 含所有特征的 3min DataFrame。
        signal: 要复盘的 Signal 对象。
        bars_before: 信号前显示多少根 bar。
        bars_after: 信号后显示多少根 bar。
        trade_result: 可选,run_backtest 中一笔交易的 dict(含 exit_price, exit_time)。

    Returns:
        matplotlib Figure。
    """
    plt.style.use("dark_background")

    # 定位信号 bar
    if signal.timestamp not in df.index:
        raise ValueError(f"信号时间戳 {signal.timestamp} 不在 DataFrame 中")
    sig_idx = df.index.get_loc(signal.timestamp)
    start = max(0, sig_idx - bars_before)
    end   = min(len(df), sig_idx + bars_after + 1)
    view  = df.iloc[start:end].copy()

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        facecolor="#1a1a2e",
    )
    for ax in [ax_price, ax_vol]:
        ax.set_facecolor("#1a1a2e")

    xs = range(len(view))
    labels = [t.strftime("%H:%M") if t.minute % 15 == 0 else ""
              for t in view.index]

    # ── K 线 ──────────────────────────────────────────────────────
    for i, (_, row) in enumerate(view.iterrows()):
        up = row["close"] >= row["open"]
        c  = "#26a69a" if up else "#ef5350"
        ax_price.plot([i, i], [row["low"], row["high"]], color=c, lw=0.8)
        ax_price.add_patch(plt.Rectangle(
            (i - 0.35, min(row["open"], row["close"])),
            0.7, max(abs(row["close"] - row["open"]), 1e-4),
            color=c,
        ))

    # ── EMA ───────────────────────────────────────────────────────
    for col, color, label in [
        ("ema_9",  "#FFD700", "EMA9"),
        ("ema_21", "#FFA500", "EMA21"),
        ("ema_50", "#4169E1", "EMA50"),
    ]:
        if col in view.columns:
            ax_price.plot(xs, view[col], color=color, lw=1.2, label=label, zorder=2)

    # ── VWAP ──────────────────────────────────────────────────────
    if "vwap" in view.columns:
        ax_price.plot(xs, view["vwap"], color="#DA70D6", lw=1.5, label="VWAP", zorder=2)
        if "vwap_upper_1s" in view.columns:
            ax_price.fill_between(
                xs, view["vwap_lower_1s"], view["vwap_upper_1s"],
                color="#DA70D6", alpha=0.1,
            )

    # ── 阻力/支撑线 ───────────────────────────────────────────────
    ctx = signal.context
    if signal.direction == "put" and "resistance_level" in ctx:
        ax_price.axhline(
            ctx["resistance_level"], color="#FF6B6B", lw=1.2,
            ls="--", label=f"Resistance {ctx['resistance_level']:.2f}",
        )
    elif signal.direction == "call" and "support_level" in ctx:
        ax_price.axhline(
            ctx["support_level"], color="#4ECDC4", lw=1.2,
            ls="--", label=f"Support {ctx['support_level']:.2f}",
        )

    # ── 入场标记 ──────────────────────────────────────────────────
    local_sig_idx = sig_idx - start
    if 0 <= local_sig_idx < len(view):
        arrow_dir = "^" if signal.direction == "call" else "v"
        arrow_y   = view["low"].iloc[local_sig_idx] - 0.1 if signal.direction == "call" \
                    else view["high"].iloc[local_sig_idx] + 0.1
        ax_price.plot(
            local_sig_idx, arrow_y,
            marker=arrow_dir, ms=14,
            color="#00FF7F" if signal.direction == "call" else "#FF4444",
            zorder=5, label=f"Entry ({signal.direction.upper()})",
        )
        # 止损线
        ax_price.axhline(signal.stop_level, color="orange", lw=0.8, ls=":", alpha=0.8)
        if signal.target_level:
            ax_price.axhline(signal.target_level, color="lime", lw=0.8, ls=":", alpha=0.8)

    # ── 出场标记 ──────────────────────────────────────────────────
    if trade_result and "exit_time" in trade_result:
        exit_ts = trade_result["exit_time"]
        if exit_ts in view.index:
            exit_local = view.index.get_loc(exit_ts)
            exit_price = trade_result["exit_price"]
            pnl        = trade_result.get("pnl_pct", 0)
            exit_color = "#26a69a" if pnl >= 0 else "#ef5350"
            ax_price.plot(
                exit_local, exit_price,
                marker="x", ms=10, mew=2,
                color=exit_color, zorder=5,
                label=f"Exit ({trade_result.get('exit_reason','?')} {pnl:+.2f}%)",
            )

    # ── 成交量 + RVOL ──────────────────────────────────────────────
    vol_colors = ["#26a69a" if view["close"].iloc[i] >= view["open"].iloc[i] else "#ef5350"
                  for i in range(len(view))]
    ax_vol.bar(xs, view["volume"], color=vol_colors, alpha=0.7)

    if "rvol" in view.columns:
        ax_vol2 = ax_vol.twinx()
        ax_vol2.plot(xs, view["rvol"], color="yellow", lw=1.0)
        ax_vol2.axhline(1.5, color="orange", lw=0.6, ls="--", alpha=0.6)
        ax_vol2.set_ylabel("RVOL", color="yellow", fontsize=8)
        ax_vol2.tick_params(colors="yellow", labelsize=7)

    # ── 信号标注框 ────────────────────────────────────────────────
    conf_str = f"Conf: {signal.confidence:.2f}"
    hold_str = f"Hold: {signal.suggested_hold_min}min"
    title = (
        f"[{signal.pattern.upper()}] {signal.direction.upper()} | "
        f"{signal.timestamp.strftime('%Y-%m-%d %H:%M')} | {conf_str} | {hold_str}"
    )
    ax_price.set_title(title, fontsize=11, color="white")
    ax_price.legend(loc="upper left", fontsize=8, facecolor="#222",
                    labelcolor="white", framealpha=0.7)
    ax_price.set_ylabel("Price (USD)", color="white")
    ax_price.tick_params(colors="white")

    ax_vol.set_ylabel("Volume", color="white", fontsize=8)
    ax_vol.tick_params(colors="white", labelsize=7)
    ax_vol.set_xticks(list(xs))
    ax_vol.set_xticklabels(labels, rotation=45, fontsize=7)

    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.tight_layout()
    return fig
