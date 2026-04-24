"""
backtest/simulator.py
---------------------
信号 → 模拟持仓 → 退出,输出每笔交易的 P&L、MFE、MAE。

入场规则:
  - 信号产生在 bar t 收盘
  - 入场价 = bar t+1 的 open + slippage(做多加,做空减)
  - slippage = slippage_bps / 10000 * entry_price

退出策略(三种,并行对比):
  - fixed_time:   持有 suggested_hold_min 后平仓
  - target_stop:  先到 target/stop 则平仓,超时按 fixed_time
  - trailing_atr: 以 1×ATR 为跟踪止损

输出 DataFrame 每行为一笔交易,含:
  entry_time, exit_time, direction, entry_price, exit_price,
  pnl_pct, mfe_pct, mae_pct, exit_reason, signal_confidence, exit_strategy
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from spx_scanner.config_loader import load_config

logger = logging.getLogger(__name__)

ExitStrategy = Literal["fixed_time", "target_stop", "trailing_atr"]


def run_backtest(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    exit_strategies: list[ExitStrategy] | None = None,
) -> pd.DataFrame:
    """对所有信号运行回测,返回交易记录 DataFrame。

    Args:
        df: 含所有特征的 3min OHLCV DataFrame。
        signals_df: scanner.engine.scan() 的输出。
        exit_strategies: 要测试的退出策略列表,None 则用 config 默认值。

    Returns:
        trades DataFrame。
    """
    cfg = load_config()
    bt = cfg["backtest"]
    slippage_bps = bt["entry"]["slippage_bps"]
    commission   = bt["entry"]["commission_per_trade_usd"]
    trailing_atr_mult = bt["trailing_atr_multiplier"]

    if exit_strategies is None:
        exit_strategies = list(bt["exit_strategies"])

    if signals_df.empty:
        return pd.DataFrame()

    all_trades = []

    for _, sig in signals_df.iterrows():
        sig_ts      = sig["timestamp"]
        direction   = sig["direction"]
        hold_min    = int(sig["suggested_hold_min"])
        stop_level  = sig["stop_level"]
        target_lvl  = sig.get("target_level", None)
        confidence  = sig["confidence"]

        # 找 bar t (信号时间)
        if sig_ts not in df.index:
            continue
        loc = df.index.get_loc(sig_ts)
        # get_loc 可能返回 slice(重复 index),取最后匹配位置
        t_idx = int(loc.stop - 1) if isinstance(loc, slice) else int(loc)
        if t_idx + 1 >= len(df):
            continue

        # 入场:bar t+1 open
        entry_bar = df.iloc[t_idx + 1]
        raw_entry = entry_bar["open"]
        slip      = slippage_bps / 10_000 * raw_entry
        entry_price = raw_entry + slip if direction == "call" else raw_entry - slip
        entry_time  = df.index[t_idx + 1]

        # 持仓区间
        hold_bars = max(1, hold_min // 3)  # 3min per bar
        exit_end_idx = min(t_idx + 1 + hold_bars, len(df) - 1)
        future_df = df.iloc[t_idx + 1 : exit_end_idx + 1]

        if future_df.empty:
            continue

        for strategy in exit_strategies:
            trade = _simulate_trade(
                direction=direction,
                entry_price=entry_price,
                entry_time=entry_time,
                future_df=future_df,
                stop_level=stop_level,
                target_level=target_lvl,
                strategy=strategy,
                trailing_atr_mult=trailing_atr_mult,
                confidence=confidence,
                pattern=sig["pattern"],
                symbol=sig.get("symbol", "SPY"),
            )
            all_trades.append(trade)

    if not all_trades:
        return pd.DataFrame()

    trades = pd.DataFrame(all_trades)
    return trades


def _simulate_trade(
    direction: str,
    entry_price: float,
    entry_time: pd.Timestamp,
    future_df: pd.DataFrame,
    stop_level: float,
    target_level: float | None,
    strategy: str,
    trailing_atr_mult: float,
    confidence: float,
    pattern: str,
    symbol: str,
) -> dict:
    """模拟一笔交易,返回交易结果 dict。"""
    sign = 1 if direction == "call" else -1

    highs  = future_df["high"].values
    lows   = future_df["low"].values
    closes = future_df["close"].values
    atrs   = future_df["atr_14"].values if "atr_14" in future_df else np.full(len(future_df), np.nan)

    exit_price  = closes[-1]
    exit_time   = future_df.index[-1]
    exit_reason = "time"
    trailing_stop = stop_level  # for trailing strategy

    # 计算每 bar 的 MFE/MAE
    if direction == "call":
        bar_high = highs
        bar_low  = lows
    else:
        bar_high = -lows   # invert for short
        bar_low  = -highs

    entry_signed = entry_price * sign

    mfe_price = entry_price
    mae_price = entry_price

    for j in range(len(future_df)):
        h, l, c = highs[j], lows[j], closes[j]
        atr = atrs[j] if not np.isnan(atrs[j]) else 0.0

        if direction == "call":
            mfe_price = max(mfe_price, h)
            mae_price = min(mae_price, l)
        else:
            mfe_price = min(mfe_price, l)
            mae_price = max(mae_price, h)

        # 检查止损
        if strategy in ("target_stop", "trailing_atr"):
            hit_stop = (direction == "call" and l <= stop_level) or \
                       (direction == "put"  and h >= stop_level)
            if hit_stop:
                exit_price  = stop_level
                exit_time   = future_df.index[j]
                exit_reason = "stop"
                break

        # 检查目标
        if strategy == "target_stop" and target_level is not None:
            hit_target = (direction == "call" and h >= target_level) or \
                         (direction == "put"  and l <= target_level)
            if hit_target:
                exit_price  = target_level
                exit_time   = future_df.index[j]
                exit_reason = "target"
                break

        # 跟踪止损更新
        if strategy == "trailing_atr" and atr > 0:
            if direction == "call":
                new_stop = c - trailing_atr_mult * atr
                trailing_stop = max(trailing_stop, new_stop)
                if l <= trailing_stop and j > 0:  # 跳过入场 bar
                    exit_price  = trailing_stop
                    exit_time   = future_df.index[j]
                    exit_reason = "trailing_stop"
                    break
            else:
                new_stop = c + trailing_atr_mult * atr
                trailing_stop = min(trailing_stop, new_stop)
                if h >= trailing_stop and j > 0:
                    exit_price  = trailing_stop
                    exit_time   = future_df.index[j]
                    exit_reason = "trailing_stop"
                    break

    # P&L(百分比)
    if direction == "call":
        pnl_pct = (exit_price - entry_price) / entry_price
        mfe_pct = (mfe_price - entry_price) / entry_price
        mae_pct = (mae_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price
        mfe_pct = (entry_price - mfe_price) / entry_price
        mae_pct = (entry_price - mae_price) / entry_price

    return {
        "symbol":        symbol,
        "pattern":       pattern,
        "direction":     direction,
        "exit_strategy": strategy,
        "entry_time":    entry_time,
        "exit_time":     exit_time,
        "entry_price":   round(entry_price, 4),
        "exit_price":    round(exit_price,  4),
        "pnl_pct":       round(pnl_pct * 100, 4),   # 百分比
        "mfe_pct":       round(mfe_pct * 100, 4),
        "mae_pct":       round(mae_pct * 100, 4),
        "exit_reason":   exit_reason,
        "confidence":    round(confidence, 3),
    }
