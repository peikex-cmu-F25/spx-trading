"""
viz/dashboard.py
----------------
Streamlit 仪表板 — 3 页面:
  Page 1: 日扫描      — 按日期查看信号、K线回顾
  Page 2: 历史统计    — 全套回测指标 + 分组分析
  Page 3: 实时信号    — (v2 占位)

启动:  streamlit run spx_scanner/viz/dashboard.py
"""

from __future__ import annotations

import sys
import os

# 确保项目根目录在 path 中
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from spx_scanner.data_layer.loader import load_data
from spx_scanner.data_layer.resampler import resample_1m_to_3m
from spx_scanner.features import compute_all_features
from spx_scanner.scanner.engine import ScannerEngine
from spx_scanner.patterns.registry import get_all_patterns
from spx_scanner.backtest.simulator import run_backtest
from spx_scanner.backtest.metrics import (
    compute_metrics, confidence_calibration, learning_curve,
)
from spx_scanner.backtest.grouping import (
    enrich_trades, group_by_segment, group_by_confidence,
    session_heatmap,
)
from spx_scanner.config_loader import load_config

# ─────────────────────────────────────────────────────────────────────────────
# 全局设置
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SPX 0DTE Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_DATA_PATH = os.path.join(_ROOT, "data", "spy_1min.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# 缓存层
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & resampling data…")
def _load_df() -> pd.DataFrame:
    df1m = load_data(_DATA_PATH)
    df3m = resample_1m_to_3m(df1m)
    return compute_all_features(df3m)


@st.cache_data(show_spinner="Scanning patterns…")
def _scan(_df: pd.DataFrame) -> pd.DataFrame:
    patterns = get_all_patterns("SPY")
    engine   = ScannerEngine(patterns=patterns)
    return engine.scan(_df)


@st.cache_data(show_spinner="Running backtest…")
def _backtest(_df: pd.DataFrame, _sigs: pd.DataFrame) -> pd.DataFrame:
    return run_backtest(_df, _sigs)


# ─────────────────────────────────────────────────────────────────────────────
# 共用工具
# ─────────────────────────────────────────────────────────────────────────────

def _make_candlestick(view: pd.DataFrame, title: str = "") -> go.Figure:
    """Plotly K线图 + EMA + VWAP。"""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=view.index,
        open=view["open"], high=view["high"],
        low=view["low"],   close=view["close"],
        name="SPY",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    ema_cfg = [("ema_9", "#FFD700", "EMA9"), ("ema_21", "#FFA500", "EMA21"),
               ("ema_50", "#4169E1", "EMA50")]
    for col, color, label in ema_cfg:
        if col in view.columns:
            fig.add_trace(go.Scatter(
                x=view.index, y=view[col],
                line=dict(color=color, width=1),
                name=label, opacity=0.8,
            ))

    if "vwap" in view.columns:
        fig.add_trace(go.Scatter(
            x=view.index, y=view["vwap"],
            line=dict(color="#DA70D6", width=1.5),
            name="VWAP",
        ))
        if "vwap_upper_1s" in view.columns:
            fig.add_trace(go.Scatter(
                x=list(view.index) + list(view.index[::-1]),
                y=list(view["vwap_upper_1s"]) + list(view["vwap_lower_1s"][::-1]),
                fill="toself", fillcolor="rgba(218,112,214,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="VWAP±1σ",
                showlegend=False,
            ))

    fig.update_layout(
        title=title, height=480,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def _add_signal_markers(fig: go.Figure, day_sigs: pd.DataFrame) -> go.Figure:
    """在 K线图上叠加信号箭头。"""
    for _, sig in day_sigs.iterrows():
        color  = "#00FF7F" if sig["direction"] == "call" else "#FF4444"
        symbol = "triangle-up" if sig["direction"] == "call" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[sig["timestamp"]], y=[sig["entry_price"]],
            mode="markers+text",
            marker=dict(symbol=symbol, size=14, color=color),
            text=[f"{sig['pattern'][:6]}<br>{sig['confidence']:.2f}"],
            textposition="top center" if sig["direction"] == "put" else "bottom center",
            textfont=dict(size=8, color=color),
            name=f"{sig['pattern']} {sig['direction']}",
            showlegend=False,
        ))
        # 止损线
        fig.add_hline(
            y=sig["stop_level"], line_dash="dot",
            line_color="orange", opacity=0.5,
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar(df: pd.DataFrame, sigs_df: pd.DataFrame) -> tuple:
    cfg = load_config()

    st.sidebar.header("SPX 0DTE Scanner")
    page = st.sidebar.radio(
        "页面",
        ["📅 日扫描", "📊 历史统计", "⚡ 实时信号(v2)"],
    )

    st.sidebar.markdown("---")

    # 日期选择
    all_dates = sorted(set(df.index.date))
    sel_date = st.sidebar.selectbox(
        "日期",
        all_dates,
        index=len(all_dates) - 1,
        format_func=str,
    )

    # Pattern 过滤
    all_patterns = sorted(sigs_df["pattern"].unique()) if not sigs_df.empty else []
    default_pats = cfg["dashboard"]["patterns_enabled_default"]
    sel_patterns = st.sidebar.multiselect(
        "Pattern",
        all_patterns,
        default=[p for p in default_pats if p in all_patterns] or all_patterns,
    )

    # 最低置信度
    min_conf = st.sidebar.slider(
        "最低置信度",
        min_value=0.50, max_value=0.90, value=cfg["dashboard"]["min_confidence_default"],
        step=0.05,
    )

    # 退出策略(历史统计页用)
    sel_strategy = st.sidebar.selectbox(
        "退出策略",
        ["fixed_time", "target_stop", "trailing_atr"],
        index=1,
    )

    return page, sel_date, sel_patterns, min_conf, sel_strategy


# ─────────────────────────────────────────────────────────────────────────────
# Page 1: 日扫描
# ─────────────────────────────────────────────────────────────────────────────

def page_daily_scan(
    df: pd.DataFrame,
    sigs_df: pd.DataFrame,
    sel_date,
    sel_patterns: list[str],
    min_conf: float,
):
    st.title(f"📅 日扫描 — {sel_date}")

    # 过滤当日 + 筛选条件
    day_df = df[df.index.date == sel_date]
    if day_df.empty:
        st.warning("该日无数据。")
        return

    day_sigs = sigs_df[
        (pd.to_datetime(sigs_df["timestamp"]).dt.date == sel_date) &
        (sigs_df["pattern"].isin(sel_patterns) if sel_patterns else True) &
        (sigs_df["confidence"] >= min_conf)
    ] if not sigs_df.empty else pd.DataFrame()

    # 统计
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总信号数", len(day_sigs))
    col2.metric("Call", int((day_sigs["direction"] == "call").sum()) if not day_sigs.empty else 0)
    col3.metric("Put",  int((day_sigs["direction"] == "put").sum())  if not day_sigs.empty else 0)
    col4.metric("最高置信度", f"{day_sigs['confidence'].max():.2f}" if not day_sigs.empty else "—")

    # K线图 + 信号标记
    st.subheader("当日 K线 (3min)")
    fig = _make_candlestick(day_df, title=f"SPY 3min — {sel_date}")
    if not day_sigs.empty:
        fig = _add_signal_markers(fig, day_sigs)
    st.plotly_chart(fig, use_container_width=True)

    # 量能子图
    if "rvol" in day_df.columns:
        vol_fig = go.Figure()
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(day_df["close"], day_df["open"])]
        vol_fig.add_trace(go.Bar(
            x=day_df.index, y=day_df["volume"],
            marker_color=colors, name="Volume", opacity=0.7,
        ))
        vol_fig.add_trace(go.Scatter(
            x=day_df.index, y=day_df["rvol"],
            yaxis="y2", line=dict(color="yellow", width=1),
            name="RVOL",
        ))
        vol_fig.update_layout(
            height=180, template="plotly_dark",
            margin=dict(l=40, r=20, t=10, b=20),
            xaxis_rangeslider_visible=False,
            yaxis2=dict(overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(vol_fig, use_container_width=True)

    # 信号列表
    st.subheader(f"信号列表 ({len(day_sigs)} 条)")
    if not day_sigs.empty:
        display = day_sigs[[
            "timestamp", "pattern", "direction", "confidence",
            "entry_price", "stop_level", "target_level", "suggested_hold_min",
        ]].copy()
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%H:%M")
        display = display.rename(columns={
            "timestamp": "时间", "pattern": "Pattern", "direction": "方向",
            "confidence": "置信度", "entry_price": "入场价",
            "stop_level": "止损", "target_level": "目标", "suggested_hold_min": "持仓min",
        })
        st.dataframe(
            display.sort_values("置信度", ascending=False),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("当日无符合条件的信号。")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2: 历史统计
# ─────────────────────────────────────────────────────────────────────────────

def page_historical(
    df: pd.DataFrame,
    sigs_df: pd.DataFrame,
    sel_strategy: str,
    min_conf: float,
):
    st.title("📊 历史统计")

    if sigs_df.empty:
        st.warning("无信号数据。")
        return

    trades_all = _backtest(df, sigs_df)
    if trades_all.empty:
        st.warning("无交易数据。")
        return

    trades = trades_all[trades_all["exit_strategy"] == sel_strategy].copy()
    trades = trades[trades["confidence"] >= min_conf]
    enriched = enrich_trades(trades, df)

    # ── 总体指标 ──────────────────────────────────────────────────────────
    st.subheader("全局指标")
    metrics = compute_metrics(
        trades.assign(exit_strategy=sel_strategy)
    )
    disp_cols = [
        "n_trades", "win_rate", "expect_pct", "profit_factor",
        "sharpe", "max_drawdown_pct", "calmar",
    ]
    st.dataframe(
        metrics[[c for c in disp_cols if c in metrics.columns]].style.format({
            "win_rate": "{:.1%}", "expect_pct": "{:.3f}%",
            "profit_factor": "{:.2f}", "sharpe": "{:.2f}",
            "max_drawdown_pct": "{:.2f}%", "calmar": "{:.2f}",
        }),
        use_container_width=True,
    )

    # ── 胜率热力图 ─────────────────────────────────────────────────────────
    st.subheader("胜率热力图 — Pattern × Session")
    hm = session_heatmap(enriched, value="win_rate")
    if not hm.empty:
        fig_hm = px.imshow(
            hm.astype(float),
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            text_auto=".2f",
            template="plotly_dark",
            height=280,
        )
        fig_hm.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── 置信度校准 ─────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("置信度校准曲线")
        calib = confidence_calibration(enriched, n_bins=5)
        if not calib.empty:
            mid = (calib["bin_low"] + calib["bin_high"]) / 2
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[calib["bin_low"].min(), calib["bin_high"].max()],
                y=[calib["bin_low"].min(), calib["bin_high"].max()],
                mode="lines", line=dict(color="gray", dash="dash"),
                name="Perfect calibration",
            ))
            fig_cal.add_trace(go.Scatter(
                x=mid, y=calib["actual_win_rate"],
                mode="markers+lines",
                marker=dict(
                    size=calib["n_trades"].apply(lambda n: max(6, min(20, n))),
                    color=calib["sufficient"].map({True: "#26a69a", False: "gray"}),
                ),
                name="Actual win rate",
            ))
            fig_cal.update_layout(
                template="plotly_dark", height=280,
                xaxis_title="Model Confidence",
                yaxis_title="Actual Win Rate",
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_cal, use_container_width=True)

    with col_r:
        st.subheader("学习曲线 (Win Rate by Time)")
        lc = learning_curve(enriched, n_blocks=5)
        if not lc.empty:
            fig_lc = go.Figure(go.Bar(
                x=[str(b)[:10] for b in lc["block_start"]],
                y=lc["win_rate"],
                marker_color=lc["win_rate"].apply(
                    lambda v: "#26a69a" if v >= 0.5 else "#ef5350"
                ),
                text=lc["win_rate"].apply(lambda v: f"{v:.0%}"),
                textposition="outside",
            ))
            fig_lc.add_hline(y=0.5, line_dash="dash", line_color="orange")
            fig_lc.update_layout(
                template="plotly_dark", height=280, yaxis_range=[0, 1],
                xaxis_title="Time Block", yaxis_title="Win Rate",
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_lc, use_container_width=True)

    # ── 累积 P&L 曲线 ──────────────────────────────────────────────────────
    st.subheader("累积 P&L 曲线 (按 Pattern)")
    fig_pnl = go.Figure()
    for pat, grp in enriched.groupby("pattern"):
        grp_sorted = grp.sort_values("entry_time")
        cum_pnl = grp_sorted["pnl_pct"].cumsum()
        fig_pnl.add_trace(go.Scatter(
            x=grp_sorted["entry_time"],
            y=cum_pnl,
            mode="lines",
            name=pat,
            line=dict(width=2),
        ))
    fig_pnl.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_pnl.update_layout(
        template="plotly_dark", height=320,
        xaxis_title="Date", yaxis_title="Cumulative P&L %",
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ── Session 分组 ───────────────────────────────────────────────────────
    st.subheader("按时段分组")
    seg_tbl = group_by_segment(enriched, by_pattern=True)
    if not seg_tbl.empty:
        st.dataframe(
            seg_tbl[["n_trades", "win_rate", "expect_pct", "profit_factor"]].style.format({
                "win_rate": "{:.1%}", "expect_pct": "{:.3f}%", "profit_factor": "{:.2f}",
            }),
            use_container_width=True,
        )

    # ── 最佳/最差 10 笔 ────────────────────────────────────────────────────
    st.subheader("最佳 / 最差 10 笔交易")
    col_best, col_worst = st.columns(2)
    with col_best:
        st.markdown("**最佳**")
        st.dataframe(
            enriched.nlargest(10, "pnl_pct")[
                ["entry_time", "pattern", "direction", "pnl_pct", "confidence"]
            ].style.format({"pnl_pct": "{:+.3f}%", "confidence": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )
    with col_worst:
        st.markdown("**最差**")
        st.dataframe(
            enriched.nsmallest(10, "pnl_pct")[
                ["entry_time", "pattern", "direction", "pnl_pct", "confidence"]
            ].style.format({"pnl_pct": "{:+.3f}%", "confidence": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

    # ── OOS 样本量检查 ─────────────────────────────────────────────────────
    st.subheader("样本量检查")
    cfg = load_config()
    min_n = cfg["backtest"]["min_samples_per_pattern"]
    cnts = enriched.groupby("pattern").size().rename("n_trades").to_frame()
    cnts["sufficient"] = cnts["n_trades"] >= min_n
    cnts["status"] = cnts["sufficient"].map({True: "✅ 充足", False: "⚠️ 不足"})
    st.dataframe(cnts, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3: 实时信号 (v2 占位)
# ─────────────────────────────────────────────────────────────────────────────

def page_live():
    st.title("⚡ 实时信号 (v2)")
    st.info(
        "实时信号功能在 v2 中开启。\n\n"
        "计划接入: yfinance 流式数据 / Polygon.io WebSocket\n\n"
        "当前可在[日扫描]页查看历史数据的逐日信号。"
    )
    st.markdown("""
    **v2 路线图:**
    - [ ] Polygon.io WebSocket 连接
    - [ ] 1min bar 增量更新
    - [ ] scan_live() 实时触发
    - [ ] Telegram / Discord webhook 推送
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 数据加载
    df = _load_df()

    # 扫描
    sigs_df = _scan(df)

    # 侧边栏
    page, sel_date, sel_patterns, min_conf, sel_strategy = _sidebar(df, sigs_df)

    # 路由
    if page == "📅 日扫描":
        page_daily_scan(df, sigs_df, sel_date, sel_patterns, min_conf)
    elif page == "📊 历史统计":
        page_historical(df, sigs_df, sel_strategy, min_conf)
    else:
        page_live()


if __name__ == "__main__":
    main()
