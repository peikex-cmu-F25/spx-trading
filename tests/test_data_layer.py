"""
tests/test_data_layer.py
------------------------
数据层单元测试:loader / resampler / validator
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

from spx_scanner.data_layer.loader import (
    _normalize_schema,
    filter_rth,
    load_data,
)
from spx_scanner.data_layer.resampler import (
    resample_1m_to_3m,
    resample_1m_to_15m,
    resample_to_timeframe,
)
from spx_scanner.data_layer.validator import validate_data

EASTERN = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_1min_day(date: str = "2025-01-02") -> pd.DataFrame:
    """生成一天完整 RTH 1min 数据(390 bar),保证 OHLC 逻辑正确。"""
    idx = pd.date_range(
        start=f"{date} 09:30",
        periods=390,
        freq="1min",
        tz="US/Eastern",
    )
    rng = np.random.default_rng(42)
    open_ = 500 + np.cumsum(rng.normal(0, 0.05, 390))
    close = open_ + rng.normal(0, 0.03, 390)
    # high >= max(open, close),  low <= min(open, close)
    high = np.maximum(open_, close) + rng.uniform(0.001, 0.05, 390)
    low = np.minimum(open_, close) - rng.uniform(0.001, 0.05, 390)
    data = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(50_000, 500_000, 390),
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def df_1min():
    return _make_1min_day()


@pytest.fixture
def df_1min_two_days():
    d1 = _make_1min_day("2025-01-02")
    d2 = _make_1min_day("2025-01-03")
    return pd.concat([d1, d2]).sort_index()


# ---------------------------------------------------------------------------
# loader 测试
# ---------------------------------------------------------------------------

class TestNormalizeSchema:
    def test_basic_columns(self, df_1min):
        df = _normalize_schema(df_1min.copy())
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_timezone_conversion(self, df_1min):
        # 给一个 UTC 时区的 df
        df_utc = df_1min.copy()
        df_utc.index = df_utc.index.tz_convert("UTC")
        result = _normalize_schema(df_utc, tz="US/Eastern")
        assert str(result.index.tz) == "US/Eastern"

    def test_missing_column_raises(self, df_1min):
        df = df_1min.drop(columns=["volume"])
        with pytest.raises(ValueError, match="缺少必要列"):
            _normalize_schema(df)

    def test_volume_is_int(self, df_1min):
        df = _normalize_schema(df_1min.copy())
        assert df["volume"].dtype == int

    def test_no_duplicate_index(self):
        idx = pd.date_range("2025-01-02 09:30", periods=5, freq="1min", tz="US/Eastern")
        idx_dup = idx.append(idx[:2])
        df = pd.DataFrame(
            {"open": 1, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100},
            index=idx_dup,
        )
        result = _normalize_schema(df)
        assert result.index.is_unique


class TestFilterRTH:
    def test_rth_only(self):
        # 包含盘前和 RTH 数据
        idx = pd.date_range("2025-01-02 08:00", periods=600, freq="1min", tz="US/Eastern")
        df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx)
        result = filter_rth(df)
        times = result.index.time
        import datetime
        assert all(t >= datetime.time(9, 30) for t in times)
        assert all(t < datetime.time(16, 0) for t in times)

    def test_full_day_count(self, df_1min):
        result = filter_rth(df_1min)
        assert len(result) == 390


class TestLoadData:
    def test_load_parquet(self, df_1min, tmp_path):
        p = tmp_path / "test.parquet"
        df_1min.to_parquet(p)
        loaded = load_data(p, rth_only=False)
        assert len(loaded) == len(df_1min)
        assert set(loaded.columns) == {"open", "high", "low", "close", "volume"}

    def test_load_csv(self, df_1min, tmp_path):
        p = tmp_path / "test.csv"
        df_1min.to_csv(p)
        loaded = load_data(p, rth_only=False)
        assert len(loaded) == len(df_1min)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path.parquet")

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "test.xlsx"
        p.touch()
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_data(p)


# ---------------------------------------------------------------------------
# resampler 测试
# ---------------------------------------------------------------------------

class TestResampler:
    def test_3min_bar_count(self, df_1min):
        df_3m = resample_1m_to_3m(df_1min)
        # 390 / 3 = 130 bars
        assert len(df_3m) == 130

    def test_15min_bar_count(self, df_1min):
        df_15m = resample_1m_to_15m(df_1min)
        # 390 / 15 = 26 bars
        assert len(df_15m) == 26

    def test_ohlc_integrity(self, df_1min):
        """重采样后 high >= low, high >= close, low <= open。"""
        df_3m = resample_1m_to_3m(df_1min)
        assert (df_3m["high"] >= df_3m["low"]).all()
        assert (df_3m["high"] >= df_3m["close"]).all()
        assert (df_3m["low"] <= df_3m["open"]).all()

    def test_volume_sum(self, df_1min):
        """3min bar 的量 = 对应 3 根 1min bar 量之和。"""
        df_3m = resample_1m_to_3m(df_1min)
        first_3m_volume = df_3m["volume"].iloc[0]
        first_3_1m_volume = df_1min["volume"].iloc[:3].sum()
        assert first_3m_volume == first_3_1m_volume

    def test_open_is_first_close(self, df_1min):
        """3min bar 的 open = 第一根 1min bar 的 open。"""
        df_3m = resample_1m_to_3m(df_1min)
        assert df_3m["open"].iloc[0] == pytest.approx(df_1min["open"].iloc[0])

    def test_bar_alignment_to_930(self, df_1min):
        """第一根 3min bar 的时间戳应为 09:30。"""
        df_3m = resample_1m_to_3m(df_1min)
        import datetime
        assert df_3m.index[0].time() == datetime.time(9, 30)

    def test_two_days_resample(self, df_1min_two_days):
        df_3m = resample_1m_to_3m(df_1min_two_days)
        assert len(df_3m) == 260  # 130 × 2

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index = pd.DatetimeIndex([], tz="US/Eastern", name="timestamp")
        result = resample_1m_to_3m(empty)
        assert result.empty

    def test_custom_timeframe(self, df_1min):
        df_5m = resample_to_timeframe(df_1min, "5min")
        # 390 / 5 = 78 bars
        assert len(df_5m) == 78




# ---------------------------------------------------------------------------
# validator 测试
# ---------------------------------------------------------------------------

class TestValidator:
    def test_valid_data_passes(self, df_1min):
        report = validate_data(df_1min, timeframe="1min")
        assert report.is_valid
        assert len(report.ohlc_violations) == 0
        assert len(report.price_anomalies) == 0

    def test_ohlc_violation_detected(self, df_1min):
        df = df_1min.copy()
        # 强制制造 high < low 的违规
        df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["low"] - 1.0
        report = validate_data(df, timeframe="1min")
        assert not report.is_valid
        assert len(report.ohlc_violations) > 0

    def test_zero_volume_detected(self, df_1min):
        df = df_1min.copy()
        df.iloc[:10, df.columns.get_loc("volume")] = 0
        report = validate_data(df, timeframe="1min")
        assert len(report.zero_volume_bars) == 10

    def test_missing_bars_detected(self, df_1min):
        # 删除一半数据模拟缺失
        df = df_1min.iloc[::2].copy()  # 每隔一根取一根
        report = validate_data(df, timeframe="1min", check_missing=True)
        assert len(report.missing_bars) > 0

    def test_trading_days_count(self, df_1min_two_days):
        report = validate_data(df_1min_two_days, timeframe="1min")
        assert report.trading_days == 2

    def test_summary_is_string(self, df_1min):
        report = validate_data(df_1min, timeframe="1min")
        s = report.summary()
        assert isinstance(s, str)
        assert "验证报告" in s
