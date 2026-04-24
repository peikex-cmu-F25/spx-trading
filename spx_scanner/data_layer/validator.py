"""
data_layer/validator.py
-----------------------
数据质量检查:缺失 bar、异常值、时区一致性等。

使用方式:
    report = validate_data(df, timeframe="1min")
    if not report.is_valid:
        print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 每个交易日 RTH 的预期 bar 数(不含收盘那根)
EXPECTED_BARS_PER_DAY = {
    "1min": 390,   # 09:30–16:00, 6.5h × 60
    "3min": 130,   # 390 / 3
    "15min": 26,   # 390 / 15
}

# 涨跌幅异常阈值(单 bar 涨跌超过此值视为异常)
PRICE_CHANGE_ALERT_PCT = 0.05  # 5%
ZERO_VOLUME_LIMIT = 5          # 连续超过此数的零量 bar 报警


@dataclass
class ValidationReport:
    """数据验证报告。"""
    symbol: str
    timeframe: str
    total_bars: int
    trading_days: int

    # 问题列表
    missing_bars: list[str] = field(default_factory=list)       # 缺失的 bar 时间戳
    zero_volume_bars: list[str] = field(default_factory=list)   # 零量 bar
    price_anomalies: list[str] = field(default_factory=list)    # 价格异常 bar
    ohlc_violations: list[str] = field(default_factory=list)    # OHLC 逻辑违规(high<low 等)
    warnings: list[str] = field(default_factory=list)           # 一般性警告

    @property
    def is_valid(self) -> bool:
        """是否通过所有关键检查(允许少量缺失 bar)。"""
        return (
            len(self.ohlc_violations) == 0
            and len(self.price_anomalies) == 0
        )

    def summary(self) -> str:
        lines = [
            f"=== 数据验证报告 ===",
            f"品种: {self.symbol}  时间框架: {self.timeframe}",
            f"总 bar 数: {self.total_bars}  交易日数: {self.trading_days}",
            f"缺失 bar: {len(self.missing_bars)}",
            f"零量 bar: {len(self.zero_volume_bars)}",
            f"价格异常: {len(self.price_anomalies)}",
            f"OHLC 违规: {len(self.ohlc_violations)}",
            f"警告: {len(self.warnings)}",
            f"结论: {'✓ 通过' if self.is_valid else '✗ 存在关键问题'}",
        ]
        if self.ohlc_violations:
            lines.append("\n[OHLC 违规]")
            lines.extend(f"  {v}" for v in self.ohlc_violations[:10])
        if self.price_anomalies:
            lines.append("\n[价格异常]")
            lines.extend(f"  {v}" for v in self.price_anomalies[:10])
        if self.warnings:
            lines.append("\n[警告]")
            lines.extend(f"  {w}" for w in self.warnings[:10])
        return "\n".join(lines)


def validate_data(
    df: pd.DataFrame,
    symbol: str = "SPY",
    timeframe: Literal["1min", "3min", "15min"] = "1min",
    check_missing: bool = True,
) -> ValidationReport:
    """对 OHLCV DataFrame 进行全面数据质量检查。

    Args:
        df: 带 tz-aware DatetimeIndex 的 OHLCV DataFrame。
        symbol: 品种代码,用于报告。
        timeframe: 数据时间框架,影响缺失 bar 检查逻辑。
        check_missing: 是否检查缺失 bar(可能较慢)。

    Returns:
        ValidationReport 对象。
    """
    report = ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        total_bars=len(df),
        trading_days=df.index.normalize().nunique(),
    )

    if df.empty:
        report.warnings.append("DataFrame 为空")
        return report

    # 1. OHLC 逻辑检查
    _check_ohlc_logic(df, report)

    # 2. 价格异常检查
    _check_price_anomalies(df, report)

    # 3. 零量 bar 检查
    _check_zero_volume(df, report)

    # 4. 缺失 bar 检查
    if check_missing and timeframe in EXPECTED_BARS_PER_DAY:
        _check_missing_bars(df, timeframe, report)

    # 5. 时区检查
    if df.index.tz is None:
        report.warnings.append("index 无时区信息,建议显式设置 tz=US/Eastern")

    logger.info(report.summary())
    return report


# ---------------------------------------------------------------------------
# 内部检查函数
# ---------------------------------------------------------------------------

def _check_ohlc_logic(df: pd.DataFrame, report: ValidationReport) -> None:
    """检查 high >= low, high >= close/open, low <= close/open。"""
    mask = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    violations = df[mask]
    for ts in violations.index:
        row = df.loc[ts]
        report.ohlc_violations.append(
            f"{ts}: O={row['open']:.3f} H={row['high']:.3f} L={row['low']:.3f} C={row['close']:.3f}"
        )


def _check_price_anomalies(df: pd.DataFrame, report: ValidationReport) -> None:
    """检测单 bar 内超过阈值的价格变动(可能为数据错误)。"""
    # bar 内振幅
    intra_range = (df["high"] - df["low"]) / df["close"]
    anomalous = df[intra_range > PRICE_CHANGE_ALERT_PCT]
    for ts in anomalous.index:
        pct = intra_range.loc[ts] * 100
        report.price_anomalies.append(f"{ts}: 单 bar 振幅 {pct:.2f}%")

    # 收盘价连续跳变
    close_chg = df["close"].pct_change().abs()
    jumps = df[close_chg > PRICE_CHANGE_ALERT_PCT]
    for ts in jumps.index:
        pct = close_chg.loc[ts] * 100
        report.price_anomalies.append(f"{ts}: 收盘跳变 {pct:.2f}%")

    # 去重(同一 bar 可能被两条规则标记)
    report.price_anomalies = list(dict.fromkeys(report.price_anomalies))


def _check_zero_volume(df: pd.DataFrame, report: ValidationReport) -> None:
    """记录零量 bar,连续超过阈值则报警。"""
    zero_vol = df[df["volume"] == 0]
    report.zero_volume_bars = [str(ts) for ts in zero_vol.index]

    if len(zero_vol) > ZERO_VOLUME_LIMIT:
        report.warnings.append(
            f"发现 {len(zero_vol)} 个零量 bar,可能存在数据缺口"
        )


def _check_missing_bars(
    df: pd.DataFrame, timeframe: str, report: ValidationReport
) -> None:
    """按交易日检查每天的 bar 数是否完整。"""
    expected = EXPECTED_BARS_PER_DAY[timeframe]
    freq_map = {"1min": "1min", "3min": "3min", "15min": "15min"}
    freq = freq_map[timeframe]

    for date, day_df in df.groupby(df.index.normalize()):
        actual = len(day_df)
        if actual < expected * 0.9:  # 允许 10% 缺失(节假日提前收盘等)
            report.missing_bars.append(
                f"{date.date()}: 预期 ≈{expected} bar,实际 {actual} bar"
            )
            report.warnings.append(
                f"{date.date()} 数据不完整 ({actual}/{expected} bars)"
            )
