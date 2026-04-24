"""
patterns/base.py
----------------
Pattern 基类 + Signal 数据结构。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class Signal:
    """单个交易信号。"""
    timestamp: pd.Timestamp          # 信号产生时间(bar t 收盘)
    symbol: str                       # 品种代码
    pattern: str                      # 触发 pattern 名称
    direction: Literal["call", "put"] # 多/空方向
    confidence: float                 # 置信度 [0, 1]
    entry_price: float                # 建议入场价(bar t+1 open 附近)
    context: dict = field(default_factory=dict)  # 触发时的关键特征快照
    suggested_hold_min: int = 15      # 建议持仓分钟数
    stop_level: float = 0.0           # 止损价格
    target_level: float | None = None # 目标价(可为 None)

    def __post_init__(self):
        self.confidence = float(max(0.0, min(1.0, self.confidence)))


class Pattern(ABC):
    """所有 pattern 的抽象基类。"""

    name: str = "base"
    default_hold_min: int = 15

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[Signal]:
        """扫描 DataFrame,返回检测到的信号列表。

        Args:
            df: 含全部特征列的 3min OHLCV DataFrame。

        Returns:
            Signal 列表(无信号时返回空列表)。
        """
        ...

    @abstractmethod
    def explain(self, signal: Signal) -> str:
        """返回信号的人类可读触发原因(用于复盘)。

        Args:
            signal: 要解释的 Signal 对象。

        Returns:
            说明字符串。
        """
        ...
