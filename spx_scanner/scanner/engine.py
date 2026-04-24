"""
scanner/engine.py
-----------------
扫描引擎:遍历 DataFrame,调用所有 pattern 检测信号。
"""

from __future__ import annotations

import logging

import pandas as pd

from spx_scanner.patterns.base import Pattern, Signal

logger = logging.getLogger(__name__)


class ScannerEngine:
    """批量/实时扫描引擎。"""

    def __init__(self, patterns: list[Pattern]):
        """初始化扫描引擎。

        Args:
            patterns: 已实例化的 Pattern 列表。
        """
        self.patterns = patterns

    def scan(self, df: pd.DataFrame) -> pd.DataFrame:
        """在完整 feature DataFrame 上运行所有 pattern。

        Args:
            df: 含所有特征列的 3min DataFrame。

        Returns:
            signals DataFrame,columns 包含 Signal 的所有字段,
            按 timestamp 升序排列。
        """
        all_signals: list[Signal] = []

        for pattern in self.patterns:
            try:
                signals = pattern.detect(df)
                all_signals.extend(signals)
                logger.info(
                    "Pattern [%s]: 检测到 %d 个信号",
                    pattern.name, len(signals),
                )
            except Exception as e:
                logger.warning("Pattern [%s] 报错: %s", pattern.name, e)

        if not all_signals:
            return pd.DataFrame(columns=[
                "timestamp", "symbol", "pattern", "direction",
                "confidence", "entry_price", "suggested_hold_min",
                "stop_level", "target_level",
            ])

        records = []
        for s in all_signals:
            rec = {
                "timestamp":         s.timestamp,
                "symbol":            s.symbol,
                "pattern":           s.pattern,
                "direction":         s.direction,
                "confidence":        s.confidence,
                "entry_price":       s.entry_price,
                "suggested_hold_min": s.suggested_hold_min,
                "stop_level":        s.stop_level,
                "target_level":      s.target_level,
                **{f"ctx_{k}": v for k, v in s.context.items()},
            }
            records.append(rec)

        result = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        logger.info("扫描完成,共 %d 个信号", len(result))
        return result

    def scan_live(
        self,
        latest_bar: pd.Series,
        history: pd.DataFrame,
    ) -> list[Signal]:
        """实时模式:只检测最新 bar 是否触发信号。

        Args:
            latest_bar: 最新 bar(含特征),index 为 timestamp。
            history: 历史特征 DataFrame(含最新 bar)。

        Returns:
            触发的 Signal 列表。
        """
        all_signals: list[Signal] = []
        for pattern in self.patterns:
            # 只返回最新 bar 的信号
            try:
                signals = pattern.detect(history)
                # 过滤最新 bar
                new = [s for s in signals if s.timestamp == latest_bar.name]
                all_signals.extend(new)
            except Exception as e:
                logger.warning("Pattern [%s] live 报错: %s", pattern.name, e)
        return all_signals
