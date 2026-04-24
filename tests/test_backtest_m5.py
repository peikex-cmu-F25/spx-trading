"""
tests/test_backtest_m5.py
--------------------------
M5 全套回测框架测试:metrics(新增指标) + grouping 模块。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spx_scanner.backtest.metrics import (
    compute_metrics,
    compute_metrics_flat,
    confidence_calibration,
    learning_curve,
    _max_consecutive,
)
from spx_scanner.backtest.grouping import (
    enrich_trades,
    group_by_segment,
    group_by_vix,
    group_by_dow,
    group_by_confidence,
    session_heatmap,
    confidence_win_rate_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_trades(n: int = 40, win_rate: float = 0.55, seed: int = 0) -> pd.DataFrame:
    """合成 trades DataFrame。"""
    rng = np.random.default_rng(seed)
    pnl = np.where(rng.random(n) < win_rate,
                   rng.uniform(0.1, 1.0, n),
                   -rng.uniform(0.1, 0.8, n))
    ts = pd.date_range("2026-04-07 09:33", periods=n, freq="15min", tz="US/Eastern")
    patterns = np.tile(["failed_breakout", "orb_breakout"], n // 2 + 1)[:n]
    return pd.DataFrame({
        "pattern":       patterns,
        "direction":     "call",
        "exit_strategy": "fixed_time",
        "entry_time":    ts,
        "exit_time":     ts + pd.Timedelta("15min"),
        "entry_price":   500.0,
        "exit_price":    500.0 + pnl,
        "pnl_pct":       pnl,
        "mfe_pct":       np.abs(pnl),
        "mae_pct":       -np.abs(pnl) * 0.5,
        "exit_reason":   "time",
        "confidence":    rng.uniform(0.5, 0.9, n),
        "symbol":        "SPY",
    })


def _make_df_with_context(n: int = 200) -> pd.DataFrame:
    """合成带环境列的特征 DataFrame。"""
    idx = pd.date_range("2026-04-07 09:30", periods=n, freq="3min", tz="US/Eastern")
    segs = np.tile(["open", "midday", "close"], n // 3 + 1)[:n]
    vixxs = np.tile(["low", "mid", "high"], n // 3 + 1)[:n]
    dows = [t.day_name() for t in idx]
    return pd.DataFrame({
        "close":           500.0,
        "session_segment": segs,
        "vix_regime":      vixxs,
        "dow":             dows,
        "is_opex":         False,
        "is_fomc":         False,
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — 新增字段
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsExtended:
    def test_compute_metrics_has_new_fields(self):
        trades = _make_trades(40)
        metrics = compute_metrics(trades)
        for col in ["max_consec_wins", "max_consec_losses", "max_drawdown_pct"]:
            assert col in metrics.columns, f"Missing column: {col}"

    def test_max_drawdown_nonpositive(self):
        trades = _make_trades(40, win_rate=0.4)
        metrics = compute_metrics(trades)
        assert (metrics["max_drawdown_pct"] <= 0).all()

    def test_max_consec_wins_positive(self):
        trades = _make_trades(40, win_rate=0.9)
        metrics = compute_metrics(trades)
        assert (metrics["max_consec_wins"] >= 1).all()

    def test_total_pnl_preserved(self):
        trades = _make_trades(20)
        metrics = compute_metrics(trades)
        assert "total_pnl_pct" in metrics.columns

    def test_compute_metrics_empty_returns_empty(self):
        assert compute_metrics(pd.DataFrame()).empty

    def test_compute_metrics_flat(self):
        trades = _make_trades(30)
        metrics = compute_metrics_flat(trades)
        assert "win_rate" in metrics.columns
        assert len(metrics) == trades["pattern"].nunique()


class TestMaxConsecutive:
    def test_all_wins(self):
        w, l = _max_consecutive(np.array([True, True, True]))
        assert w == 3
        assert l == 0

    def test_all_losses(self):
        w, l = _max_consecutive(np.array([False, False, False]))
        assert w == 0
        assert l == 3

    def test_alternating(self):
        w, l = _max_consecutive(np.array([True, False, True, False]))
        assert w == 1
        assert l == 1

    def test_streak_pattern(self):
        w, l = _max_consecutive(np.array([True, True, False, False, False, True]))
        assert w == 2
        assert l == 3

    def test_empty(self):
        w, l = _max_consecutive(np.array([], dtype=bool))
        assert w == 0
        assert l == 0


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — confidence_calibration
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceCalibration:
    def test_returns_dataframe(self):
        trades = _make_trades(60)
        calib = confidence_calibration(trades, n_bins=5)
        assert isinstance(calib, pd.DataFrame)

    def test_columns_present(self):
        trades = _make_trades(60)
        calib = confidence_calibration(trades, n_bins=5)
        for col in ["bin_low", "bin_high", "n_trades", "actual_win_rate", "sufficient"]:
            assert col in calib.columns

    def test_n_bins_count(self):
        trades = _make_trades(60)
        calib = confidence_calibration(trades, n_bins=4)
        assert len(calib) == 4

    def test_win_rate_between_0_and_1(self):
        trades = _make_trades(60)
        calib = confidence_calibration(trades)
        wr = calib["actual_win_rate"].dropna()
        assert (wr >= 0).all() and (wr <= 1).all()

    def test_empty_trades_returns_empty(self):
        result = confidence_calibration(pd.DataFrame())
        assert result.empty

    def test_n_trades_sums_to_total(self):
        trades = _make_trades(60)
        calib = confidence_calibration(trades, n_bins=5)
        assert calib["n_trades"].sum() == len(trades)


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — learning_curve
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningCurve:
    def test_returns_dataframe(self):
        trades = _make_trades(50)
        lc = learning_curve(trades, n_blocks=5)
        assert isinstance(lc, pd.DataFrame)

    def test_columns_present(self):
        trades = _make_trades(50)
        lc = learning_curve(trades, n_blocks=5)
        for col in ["block_start", "block_end", "n_trades", "win_rate", "total_pnl_pct"]:
            assert col in lc.columns

    def test_n_blocks_approx(self):
        trades = _make_trades(50)
        lc = learning_curve(trades, n_blocks=5)
        # 最多 n_blocks+1 个 block(整除余数)
        assert len(lc) <= 6

    def test_win_rate_valid(self):
        trades = _make_trades(50)
        lc = learning_curve(trades)
        assert (lc["win_rate"] >= 0).all()
        assert (lc["win_rate"] <= 1).all()

    def test_empty_trades(self):
        result = learning_curve(pd.DataFrame())
        assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# grouping.py — enrich_trades
# ─────────────────────────────────────────────────────────────────────────────

class TestEnrichTrades:
    def test_adds_session_segment(self):
        trades = _make_trades(20)
        df = _make_df_with_context(200)
        enriched = enrich_trades(trades, df)
        assert "session_segment" in enriched.columns

    def test_adds_vix_regime(self):
        trades = _make_trades(20)
        df = _make_df_with_context(200)
        enriched = enrich_trades(trades, df)
        assert "vix_regime" in enriched.columns

    def test_adds_dow(self):
        trades = _make_trades(20)
        df = _make_df_with_context(200)
        enriched = enrich_trades(trades, df)
        assert "dow" in enriched.columns

    def test_adds_confidence_bucket(self):
        trades = _make_trades(20)
        df = _make_df_with_context(200)
        enriched = enrich_trades(trades, df)
        assert "confidence_bucket" in enriched.columns

    def test_empty_trades_passthrough(self):
        df = _make_df_with_context(50)
        enriched = enrich_trades(pd.DataFrame(), df)
        assert enriched.empty

    def test_no_context_cols_passthrough(self):
        trades = _make_trades(10)
        df = pd.DataFrame({"close": [500.0]})
        enriched = enrich_trades(trades, df)
        assert len(enriched) == len(trades)


# ─────────────────────────────────────────────────────────────────────────────
# grouping.py — group_by_* functions
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupBy:
    def _get_enriched(self, n: int = 60) -> pd.DataFrame:
        trades = _make_trades(n)
        df = _make_df_with_context(300)
        return enrich_trades(trades, df)

    def test_group_by_segment_returns_df(self):
        enriched = self._get_enriched()
        result = group_by_segment(enriched)
        assert isinstance(result, pd.DataFrame)

    def test_group_by_segment_has_win_rate(self):
        enriched = self._get_enriched()
        result = group_by_segment(enriched)
        assert "win_rate" in result.columns

    def test_group_by_vix_returns_df(self):
        enriched = self._get_enriched()
        result = group_by_vix(enriched)
        assert isinstance(result, pd.DataFrame)

    def test_group_by_dow_returns_df(self):
        enriched = self._get_enriched()
        result = group_by_dow(enriched)
        assert isinstance(result, pd.DataFrame)

    def test_group_by_confidence_returns_df(self):
        enriched = self._get_enriched()
        result = group_by_confidence(enriched)
        assert isinstance(result, pd.DataFrame)

    def test_group_by_segment_by_pattern(self):
        enriched = self._get_enriched()
        result = group_by_segment(enriched, by_pattern=True)
        assert isinstance(result, pd.DataFrame)
        # index should be MultiIndex (segment, pattern)
        assert result.index.nlevels == 2

    def test_group_by_win_rate_range(self):
        enriched = self._get_enriched()
        for fn in [group_by_segment, group_by_vix, group_by_dow, group_by_confidence]:
            result = fn(enriched)
            if not result.empty:
                assert (result["win_rate"] >= 0).all()
                assert (result["win_rate"] <= 1).all()

    def test_empty_trades_returns_empty(self):
        for fn in [group_by_segment, group_by_vix, group_by_dow, group_by_confidence]:
            assert fn(pd.DataFrame()).empty


# ─────────────────────────────────────────────────────────────────────────────
# grouping.py — heatmap functions
# ─────────────────────────────────────────────────────────────────────────────

class TestHeatmaps:
    def _get_enriched(self, n: int = 80) -> pd.DataFrame:
        trades = _make_trades(n)
        df = _make_df_with_context(400)
        return enrich_trades(trades, df)

    def test_session_heatmap_shape(self):
        enriched = self._get_enriched()
        hm = session_heatmap(enriched)
        assert isinstance(hm, pd.DataFrame)
        # rows = patterns, cols = session segments
        assert hm.shape[0] >= 1
        assert hm.shape[1] >= 1

    def test_session_heatmap_values_range(self):
        enriched = self._get_enriched()
        hm = session_heatmap(enriched, value="win_rate")
        vals = hm.values.flatten()
        valid = vals[~np.isnan(vals)]
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_session_heatmap_empty(self):
        assert session_heatmap(pd.DataFrame()).empty

    def test_confidence_win_rate_table(self):
        enriched = self._get_enriched()
        tbl = confidence_win_rate_table(enriched)
        assert isinstance(tbl, pd.DataFrame)

    def test_confidence_win_rate_valid_range(self):
        enriched = self._get_enriched()
        tbl = confidence_win_rate_table(enriched)
        if not tbl.empty:
            vals = tbl.values.flatten()
            valid = vals[~np.isnan(vals.astype(float))]
            assert (valid >= 0).all() and (valid <= 1).all()
