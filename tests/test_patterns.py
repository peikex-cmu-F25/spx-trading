"""
tests/test_patterns.py
----------------------
patterns/base + failed_breakout + scanner 单元测试
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import pytest

from spx_scanner.patterns.base import Signal, Pattern


# ── 公共 fixture ──────────────────────────────────────────────────────────────

def _make_feature_df(n_days: int = 15, seed: int = 0) -> pd.DataFrame:
    """生成带完整特征的 DataFrame,足够触发 FailedBreakout。"""
    from spx_scanner.data_layer.resampler import resample_to_timeframe
    from spx_scanner.features import compute_all_features

    rng = np.random.default_rng(seed)
    frames = []
    base = datetime.date(2025, 1, 2)
    for d in range(n_days):
        date = base + datetime.timedelta(days=d)
        while date.weekday() >= 5:
            date += datetime.timedelta(days=1)
        idx = pd.date_range(f"{date} 09:30", periods=130, freq="3min", tz="US/Eastern")
        o = 500 + np.cumsum(rng.normal(0, 0.05, 130))
        c = o + rng.normal(0, 0.03, 130)
        h = np.maximum(o, c) + rng.uniform(0.001, 0.05, 130)
        l = np.minimum(o, c) - rng.uniform(0.001, 0.05, 130)
        frames.append(pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c,
             "volume": rng.integers(100_000, 500_000, 130)},
            index=idx,
        ))
    df_raw = pd.concat(frames).sort_index()
    df_raw.index.name = "timestamp"
    return compute_all_features(df_raw)


@pytest.fixture(scope="module")
def df_features():
    return _make_feature_df(15)


# ── Signal dataclass ─────────────────────────────────────────────────────────

class TestSignal:
    def test_confidence_clipped(self):
        s = Signal(
            timestamp=pd.Timestamp("2025-01-02 09:30", tz="US/Eastern"),
            symbol="SPY", pattern="test", direction="call",
            confidence=1.5, entry_price=500.0,
        )
        assert s.confidence == 1.0

    def test_confidence_clipped_low(self):
        s = Signal(
            timestamp=pd.Timestamp("2025-01-02 09:30", tz="US/Eastern"),
            symbol="SPY", pattern="test", direction="put",
            confidence=-0.3, entry_price=500.0,
        )
        assert s.confidence == 0.0

    def test_direction_values(self):
        for d in ["call", "put"]:
            s = Signal(
                timestamp=pd.Timestamp("2025-01-02 09:30", tz="US/Eastern"),
                symbol="SPY", pattern="test", direction=d,
                confidence=0.7, entry_price=500.0,
            )
            assert s.direction == d


# ── FailedBreakout pattern ───────────────────────────────────────────────────

class TestFailedBreakout:
    def test_detect_returns_list(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        pattern = FailedBreakout()
        signals = pattern.detect(df_features)
        assert isinstance(signals, list)

    def test_signals_are_signal_objects(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        pattern = FailedBreakout()
        signals = pattern.detect(df_features)
        for s in signals:
            assert isinstance(s, Signal)

    def test_direction_valid(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        signals = FailedBreakout().detect(df_features)
        for s in signals:
            assert s.direction in ("call", "put")

    def test_confidence_in_range(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        signals = FailedBreakout().detect(df_features)
        for s in signals:
            assert 0.0 <= s.confidence <= 1.0

    def test_stop_level_beyond_entry(self, df_features):
        """Put 止损 > 入场价;Call 止损 < 入场价。"""
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        signals = FailedBreakout().detect(df_features)
        for s in signals:
            if s.direction == "put":
                assert s.stop_level > s.entry_price, f"Put 止损应 > 入场: {s}"
            else:
                assert s.stop_level < s.entry_price, f"Call 止损应 < 入场: {s}"

    def test_target_level_on_right_side(self, df_features):
        """Put target < entry; Call target > entry。"""
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        signals = FailedBreakout().detect(df_features)
        for s in signals:
            if s.target_level is not None:
                if s.direction == "put":
                    assert s.target_level < s.entry_price
                else:
                    assert s.target_level > s.entry_price

    def test_timestamps_in_df(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        signals = FailedBreakout().detect(df_features)
        for s in signals:
            assert s.timestamp in df_features.index

    def test_explain_returns_string(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        pattern = FailedBreakout()
        signals = pattern.detect(df_features)
        if signals:
            text = pattern.explain(signals[0])
            assert isinstance(text, str)
            assert len(text) > 10

    def test_missing_features_raises(self):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        df_bad = pd.DataFrame({"open": [1], "close": [1]})
        with pytest.raises(ValueError, match="缺少特征列"):
            FailedBreakout().detect(df_bad)


# ── ScannerEngine ────────────────────────────────────────────────────────────

class TestScannerEngine:
    def test_scan_returns_dataframe(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        from spx_scanner.scanner.engine import ScannerEngine
        engine = ScannerEngine([FailedBreakout()])
        result = engine.scan(df_features)
        assert isinstance(result, pd.DataFrame)

    def test_scan_columns(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        from spx_scanner.scanner.engine import ScannerEngine
        engine = ScannerEngine([FailedBreakout()])
        result = engine.scan(df_features)
        for col in ["timestamp", "pattern", "direction", "confidence", "entry_price"]:
            assert col in result.columns

    def test_scan_sorted_by_timestamp(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        from spx_scanner.scanner.engine import ScannerEngine
        engine = ScannerEngine([FailedBreakout()])
        result = engine.scan(df_features)
        if len(result) > 1:
            assert (result["timestamp"].diff().dropna() >= pd.Timedelta(0)).all()

    def test_scan_empty_patterns(self, df_features):
        from spx_scanner.scanner.engine import ScannerEngine
        engine = ScannerEngine([])
        result = engine.scan(df_features)
        assert result.empty

    def test_scan_confidence_range(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        from spx_scanner.scanner.engine import ScannerEngine
        engine = ScannerEngine([FailedBreakout()])
        result = engine.scan(df_features)
        if not result.empty:
            assert (result["confidence"] >= 0).all()
            assert (result["confidence"] <= 1).all()


# ── Backtest simulator ────────────────────────────────────────────────────────

class TestSimulator:
    @pytest.fixture(scope="class")
    def signals_and_trades(self, df_features):
        from spx_scanner.patterns.failed_breakout import FailedBreakout
        from spx_scanner.scanner.engine import ScannerEngine
        from spx_scanner.backtest.simulator import run_backtest
        engine = ScannerEngine([FailedBreakout()])
        signals_df = engine.scan(df_features)
        trades = run_backtest(df_features, signals_df)
        return signals_df, trades

    def test_trades_is_dataframe(self, signals_and_trades):
        _, trades = signals_and_trades
        assert isinstance(trades, pd.DataFrame)

    def test_trades_per_signal(self, signals_and_trades):
        """每个信号 × 3 退出策略 = 总交易数(除去边界信号)。"""
        signals_df, trades = signals_and_trades
        if not signals_df.empty and not trades.empty:
            n_strategies = 3
            assert len(trades) <= len(signals_df) * n_strategies

    def test_exit_strategy_values(self, signals_and_trades):
        _, trades = signals_and_trades
        if not trades.empty:
            valid = {"fixed_time", "target_stop", "trailing_atr"}
            assert trades["exit_strategy"].isin(valid).all()

    def test_exit_reason_values(self, signals_and_trades):
        _, trades = signals_and_trades
        if not trades.empty:
            valid = {"time", "stop", "target", "trailing_stop"}
            assert trades["exit_reason"].isin(valid).all()

    def test_mfe_non_negative(self, signals_and_trades):
        _, trades = signals_and_trades
        if not trades.empty:
            assert (trades["mfe_pct"] >= -0.01).all()  # allow tiny float error

    def test_mae_non_positive(self, signals_and_trades):
        _, trades = signals_and_trades
        if not trades.empty:
            assert (trades["mae_pct"] <= 0.01).all()

    def test_metrics_computes(self, signals_and_trades):
        from spx_scanner.backtest.metrics import compute_metrics
        _, trades = signals_and_trades
        if not trades.empty:
            metrics = compute_metrics(trades)
            assert not metrics.empty
            assert "win_rate" in metrics.columns
            assert (metrics["win_rate"] >= 0).all()
            assert (metrics["win_rate"] <= 1).all()
