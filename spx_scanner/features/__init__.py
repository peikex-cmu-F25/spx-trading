"""
features 包
-----------
对外暴露 compute_all_features(df) 作为主入口。
"""

from .structure import compute_all_structure
from .trend import compute_all_trend
from .volatility import compute_all_volatility
from .volume import compute_all_volume
from .momentum import compute_all_momentum
from .regime import compute_all_regime

import pandas as pd


def compute_all_features(
    df: pd.DataFrame,
    vix_series=None,
    events_path=None,
) -> pd.DataFrame:
    """在 3min OHLCV DataFrame 上计算全部特征。"""
    out = compute_all_structure(df)
    out = compute_all_trend(out)
    out = compute_all_volatility(out)
    out = compute_all_volume(out)
    out = compute_all_momentum(out)
    out = compute_all_regime(out, vix_series=vix_series, events_path=events_path)
    return out


__all__ = [
    "compute_all_features",
    "compute_all_structure",
    "compute_all_trend",
    "compute_all_volatility",
    "compute_all_volume",
    "compute_all_momentum",
    "compute_all_regime",
]
