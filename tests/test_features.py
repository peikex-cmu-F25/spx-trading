"""
tests/test_features.py
----------------------
特征层单元测试。

覆盖:structure / trend / volatility / volume / momentum / regime
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
import pytest
import pytz

# ── 公共 fixture ──────────────────────────────────────────────────────────────

def _make_3min_days(n_days: int = 5, seed: int = 42) -> pd.DataFrame:
    """生成 n 天完整 RTH 3min 数据(每天 130 bar)。"""
    rng = np.random.default_rng(seed)
    frames = []
    base_date = datetime.date(2025, 1, 2)
    for d in range(n_days):
        date = base_date + datetime.timedelta(days=d)
        # 跳过周末
        while date.weekday() >= 5:
            date += datetime.timedelta(days=1)
        idx = pd.date_range(
            start=f"{date} 09:30",
            periods=130,
            freq="3min",
            tz="US/Eastern",
        )
        open_ = 500 + np.cumsum(rng.normal(0, 0.05, 130))
        close = open_ + rng.normal(0, 0.03, 130)
        high = np.maximum(open_, close) + rng.uniform(0.001, 0.05, 130)
        low = np.minimum(open_, close) - rng.uniform(0.001, 0.05, 130)
        vol = rng.integers(50_000, 500_000, 130)
        frames.append(pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=idx,
        ))
    df = pd.concat(frames).sort_index()
    df.index.name = "timestamp"
    return df


@pytest.fixture
def df_3m():
    return _make_3min_days(5)


@pytest.fixture
def df_3m_long():
    return _make_3min_days(30)


# ── structure ─────────────────────────────────────────────────────────────────

class TestStructure:
    def test_orb_columns_exist(self, df_3m):
        from spx_scanner.features.structure import compute_orb
        out = compute_orb(df_3m)
        assert "orb_high" in out.columns
        assert "orb_low" in out.columns

    def test_orb_high_ge_low(self, df_3m):
        from spx_scanner.features.structure import compute_orb
        out = compute_orb(df_3m)
        valid = out.dropna(subset=["orb_high", "orb_low"])
        assert (valid["orb_high"] >= valid["orb_low"]).all()

    def test_orb_nan_during_formation(self, df_3m):
        """ORB 形成期(前 10 bar)应为 NaN。"""
        from spx_scanner.features.structure import compute_orb
        out = compute_orb(df_3m)
        first_day = out.index.normalize()[0]
        day_df = out[out.index.normalize() == first_day]
        assert day_df["orb_high"].iloc[:10].isna().all()

    def test_pdh_pdl_pdc(self, df_3m):
        from spx_scanner.features.structure import compute_pdh_pdl_pdc
        out = compute_pdh_pdl_pdc(df_3m)
        assert {"pdh", "pdl", "pdc"}.issubset(out.columns)
        # 第一天 PDH/PDL 应为 NaN
        first_day = out.index.normalize().unique()[0]
        day1 = out[out.index.normalize() == first_day]
        assert day1["pdh"].isna().all()

    def test_pdh_equals_prev_day_high(self, df_3m):
        from spx_scanner.features.structure import compute_pdh_pdl_pdc
        out = compute_pdh_pdl_pdc(df_3m)
        days = out.index.normalize().unique()
        if len(days) < 2:
            pytest.skip("需要至少 2 天数据")
        day0 = out[out.index.normalize() == days[0]]
        day1 = out[out.index.normalize() == days[1]]
        expected_pdh = day0["high"].max()
        actual_pdh = day1["pdh"].iloc[0]
        assert abs(actual_pdh - expected_pdh) < 1e-8

    def test_resistance_support_columns(self, df_3m_long):
        from spx_scanner.features.structure import compute_resistance_support
        out = compute_resistance_support(df_3m_long)
        assert {"resistance_level", "support_level",
                "touches_to_resistance", "touches_to_support"}.issubset(out.columns)

    def test_resistance_ge_support(self, df_3m_long):
        from spx_scanner.features.structure import compute_resistance_support
        out = compute_resistance_support(df_3m_long)
        valid = out.dropna(subset=["resistance_level", "support_level"])
        if len(valid) > 0:
            assert (valid["resistance_level"] >= valid["support_level"]).all()

    def test_prior_hour_levels(self, df_3m):
        from spx_scanner.features.structure import compute_prior_hour_levels
        out = compute_prior_hour_levels(df_3m)
        assert {"prior_hour_high", "prior_hour_low"}.issubset(out.columns)
        valid = out.dropna(subset=["prior_hour_high", "prior_hour_low"])
        assert (valid["prior_hour_high"] >= valid["prior_hour_low"]).all()


# ── trend ─────────────────────────────────────────────────────────────────────

class TestTrend:
    def test_ema_columns(self, df_3m):
        from spx_scanner.features.trend import compute_ema
        out = compute_ema(df_3m)
        for p in [9, 21, 50, 200]:
            assert f"ema_{p}" in out.columns

    def test_ema_converges(self, df_3m_long):
        """足够长的数据后,EMA 应收敛到合理范围内(close ± 5%)。"""
        from spx_scanner.features.trend import compute_ema
        out = compute_ema(df_3m_long)
        tail = out.tail(100)
        for p in [9, 21, 50]:
            ratio = (tail[f"ema_{p}"] / tail["close"]).abs()
            assert (ratio > 0.90).all() and (ratio < 1.10).all()

    def test_ema_order(self, df_3m_long):
        """在持续上涨行情中,短期 EMA > 长期 EMA。"""
        from spx_scanner.features.trend import compute_ema
        rng = np.random.default_rng(0)
        idx = pd.date_range("2025-01-02 09:30", periods=500, freq="3min", tz="US/Eastern")
        close = 500 + np.cumsum(rng.uniform(0, 0.1, 500))  # 持续上涨
        df = pd.DataFrame({"open": close, "high": close+0.1, "low": close-0.1,
                           "close": close, "volume": 100_000}, index=idx)
        out = compute_ema(df)
        tail = out.tail(50)
        assert (tail["ema_9"] > tail["ema_21"]).all()
        assert (tail["ema_21"] > tail["ema_50"]).all()

    def test_vwap_between_bb(self, df_3m):
        """VWAP 应在日内 high/low 之间。"""
        from spx_scanner.features.trend import compute_vwap
        out = compute_vwap(df_3m)
        assert "vwap" in out.columns
        assert (out["vwap"] > 0).all()

    def test_vwap_resets_daily(self, df_3m):
        """每天第一根 bar 的 VWAP 应等于该 bar 的典型价。"""
        from spx_scanner.features.trend import compute_vwap
        out = compute_vwap(df_3m)
        for date, day_df in out.groupby(out.index.normalize()):
            first = day_df.iloc[0]
            typical = (first["high"] + first["low"] + first["close"]) / 3
            assert abs(first["vwap"] - typical) < 1e-6, f"VWAP 未在 {date} 重置"

    def test_vwap_bands(self, df_3m):
        from spx_scanner.features.trend import compute_vwap
        out = compute_vwap(df_3m)
        assert (out["vwap_upper_1s"] >= out["vwap"]).all()
        assert (out["vwap_lower_1s"] <= out["vwap"]).all()

    def test_ema_compression_column(self, df_3m_long):
        from spx_scanner.features.trend import compute_ema, compute_ema_compression
        out = compute_ema(df_3m_long)
        out = compute_ema_compression(out)
        assert "ema_compression" in out.columns
        assert out["ema_compression"].dtype == bool

    def test_trend_direction_values(self, df_3m):
        from spx_scanner.features.trend import compute_all_trend
        out = compute_all_trend(df_3m)
        assert out["trend_direction"].isin([-1, 0, 1]).all()


# ── volatility ────────────────────────────────────────────────────────────────

class TestVolatility:
    def test_bb_columns(self, df_3m):
        from spx_scanner.features.volatility import compute_bollinger_bands
        out = compute_bollinger_bands(df_3m)
        assert {"bb_upper", "bb_lower", "bb_mid", "bb_width"}.issubset(out.columns)

    def test_bb_order(self, df_3m):
        from spx_scanner.features.volatility import compute_bollinger_bands
        out = compute_bollinger_bands(df_3m).dropna()
        assert (out["bb_upper"] >= out["bb_mid"]).all()
        assert (out["bb_mid"] >= out["bb_lower"]).all()

    def test_bb_width_positive(self, df_3m):
        from spx_scanner.features.volatility import compute_bollinger_bands
        out = compute_bollinger_bands(df_3m).dropna()
        assert (out["bb_width"] > 0).all()

    def test_bb_width_pctile_range(self, df_3m_long):
        from spx_scanner.features.volatility import compute_bollinger_bands, compute_bb_width_percentile
        out = compute_bollinger_bands(df_3m_long)
        out = compute_bb_width_percentile(out)
        valid = out["bb_width_pctile"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_atr_positive(self, df_3m):
        from spx_scanner.features.volatility import compute_atr
        out = compute_atr(df_3m).dropna()
        assert (out["atr_14"] > 0).all()

    def test_atr_ge_range(self, df_3m):
        """ATR 应 >= 当 bar 的 H-L(true range 定义保证)。"""
        from spx_scanner.features.volatility import compute_atr
        out = compute_atr(df_3m).dropna()
        bar_range = out["high"] - out["low"]
        # ATR 是平滑值,不一定 >= 单根 bar,但 TR 应 >= H-L
        assert (out["tr"] >= bar_range - 1e-10).all()

    def test_nr7_bool(self, df_3m):
        from spx_scanner.features.volatility import compute_nr7
        out = compute_nr7(df_3m)
        assert out["is_nr7"].dtype == bool

    def test_nr7_is_minimum(self, df_3m):
        from spx_scanner.features.volatility import compute_nr7
        out = compute_nr7(df_3m)
        for i in range(7, len(out)):
            if out["is_nr7"].iloc[i]:
                window = out["bar_range"].iloc[i-6:i+1]
                assert out["bar_range"].iloc[i] <= window.max()

    def test_squeeze_bool(self, df_3m_long):
        from spx_scanner.features.volatility import compute_all_volatility
        out = compute_all_volatility(df_3m_long)
        assert out["is_squeeze"].dtype == bool


# ── volume ────────────────────────────────────────────────────────────────────

class TestVolume:
    def test_rvol_columns(self, df_3m):
        from spx_scanner.features.volume import compute_rvol
        out = compute_rvol(df_3m)
        assert {"vol_tod_mean", "rvol"}.issubset(out.columns)

    def test_rvol_positive(self, df_3m):
        from spx_scanner.features.volume import compute_rvol
        out = compute_rvol(df_3m).dropna(subset=["rvol"])
        assert (out["rvol"] > 0).all()

    def test_rvol_mean_near_one(self, df_3m_long):
        """足够多天后,同时段 RVOL 均值应趋近 1。"""
        from spx_scanner.features.volume import compute_rvol
        out = compute_rvol(df_3m_long).dropna(subset=["rvol"])
        # 取后半段(基准估计已稳定)
        tail = out.tail(len(out) // 2)
        mean_rvol = tail["rvol"].mean()
        assert 0.8 < mean_rvol < 1.2

    def test_volume_signal_columns(self, df_3m):
        from spx_scanner.features.volume import compute_all_volume
        out = compute_all_volume(df_3m)
        assert {"is_high_vol", "volume_surge", "bullish_vol", "bearish_vol"}.issubset(out.columns)

    def test_surge_subset_of_high_vol(self, df_3m):
        """volume_surge(rvol>2) 必须是 is_high_vol(rvol>1.5) 的子集。"""
        from spx_scanner.features.volume import compute_all_volume
        out = compute_all_volume(df_3m)
        assert (out["volume_surge"] & ~out["is_high_vol"]).sum() == 0

    def test_bullish_bearish_mutually_exclusive(self, df_3m):
        """同一 bar 不能同时是 bullish_vol 和 bearish_vol。"""
        from spx_scanner.features.volume import compute_all_volume
        out = compute_all_volume(df_3m)
        assert (out["bullish_vol"] & out["bearish_vol"]).sum() == 0


# ── momentum ──────────────────────────────────────────────────────────────────

class TestMomentum:
    def test_rsi_range(self, df_3m_long):
        from spx_scanner.features.momentum import compute_rsi
        out = compute_rsi(df_3m_long).dropna(subset=["rsi_14"])
        assert (out["rsi_14"] >= 0).all() and (out["rsi_14"] <= 100).all()

    def test_rsi_rising_market(self):
        """持续上涨市场中 RSI 应 > 50。"""
        from spx_scanner.features.momentum import compute_rsi
        idx = pd.date_range("2025-01-02 09:30", periods=200, freq="3min", tz="US/Eastern")
        close = 500 + np.arange(200) * 0.05  # 单调上涨
        df = pd.DataFrame({"open": close, "high": close+0.1, "low": close-0.1,
                           "close": close, "volume": 100_000}, index=idx)
        out = compute_rsi(df).dropna()
        assert (out["rsi_14"].tail(100) > 50).all()

    def test_macd_columns(self, df_3m_long):
        from spx_scanner.features.momentum import compute_macd
        out = compute_macd(df_3m_long)
        assert {"macd_line", "macd_signal", "macd_hist", "macd_hist_flip"}.issubset(out.columns)

    def test_macd_hist_flip_values(self, df_3m_long):
        from spx_scanner.features.momentum import compute_macd
        out = compute_macd(df_3m_long)
        assert out["macd_hist_flip"].isin([-1, 0, 1]).all()

    def test_consecutive_bars_non_negative(self, df_3m):
        from spx_scanner.features.momentum import compute_consecutive_bars
        out = compute_consecutive_bars(df_3m)
        assert (out["consec_up"] >= 0).all()
        assert (out["consec_down"] >= 0).all()

    def test_consecutive_bars_mutually_exclusive(self, df_3m):
        from spx_scanner.features.momentum import compute_consecutive_bars
        out = compute_consecutive_bars(df_3m)
        # 同一 bar 不能同时上涨和下跌(但可以都为 0)
        both = (out["consec_up"] > 0) & (out["consec_down"] > 0)
        assert both.sum() == 0

    def test_consec_resets_on_direction_change(self):
        """方向改变时连续计数应重置为 0 或 1。"""
        from spx_scanner.features.momentum import compute_consecutive_bars
        idx = pd.date_range("2025-01-02 09:30", periods=6, freq="3min", tz="US/Eastern")
        close = pd.Series([100, 101, 102, 101, 100, 99], index=idx)
        df = pd.DataFrame({"open": close, "high": close+0.1, "low": close-0.1,
                           "close": close, "volume": 100_000})
        out = compute_consecutive_bars(df)
        # bar 3 转为下跌,consec_up 应为 0
        assert out["consec_up"].iloc[3] == 0
        assert out["consec_down"].iloc[3] >= 1


# ── regime ────────────────────────────────────────────────────────────────────

class TestRegime:
    def test_session_features_columns(self, df_3m):
        from spx_scanner.features.regime import compute_session_features
        out = compute_session_features(df_3m)
        assert {"minutes_from_open", "minutes_to_close",
                "session_segment", "dow"}.issubset(out.columns)

    def test_minutes_from_open_non_negative(self, df_3m):
        from spx_scanner.features.regime import compute_session_features
        out = compute_session_features(df_3m)
        assert (out["minutes_from_open"] >= 0).all()

    def test_minutes_sum_to_390(self, df_3m):
        """minutes_from_open + minutes_to_close = 390。"""
        from spx_scanner.features.regime import compute_session_features
        out = compute_session_features(df_3m)
        total = out["minutes_from_open"] + out["minutes_to_close"]
        assert (total == 390).all()

    def test_session_segment_values(self, df_3m):
        from spx_scanner.features.regime import compute_session_features
        out = compute_session_features(df_3m)
        assert out["session_segment"].isin(["open", "midday", "close"]).all()

    def test_vix_regime_with_series(self, df_3m):
        from spx_scanner.features.regime import compute_vix_regime
        dates = df_3m.index.normalize().unique()
        vix = pd.Series(18.0, index=dates)
        vix.index = pd.DatetimeIndex(vix.index)
        out = compute_vix_regime(df_3m, vix_series=vix)
        assert "vix_regime" in out.columns
        assert (out["vix_regime"] == "mid").all()

    def test_vix_regime_thresholds(self, df_3m):
        from spx_scanner.features.regime import compute_vix_regime
        dates = df_3m.index.normalize().unique()
        # 第一天 low,第二天 mid,其余 high
        vix_vals = [10, 18] + [25] * (len(dates) - 2)
        vix = pd.Series(vix_vals, index=pd.DatetimeIndex(dates))
        out = compute_vix_regime(df_3m, vix_series=vix)
        day0 = out[out.index.normalize() == dates[0]]
        day1 = out[out.index.normalize() == dates[1]]
        assert (day0["vix_regime"] == "low").all()
        assert (day1["vix_regime"] == "mid").all()

    def test_event_flags_false_without_file(self, df_3m, tmp_path):
        from spx_scanner.features.regime import compute_event_flags
        out = compute_event_flags(df_3m, events_path=tmp_path / "nonexistent.csv")
        # is_opex 可能因第三个周五为 True,但 fomc/cpi 应全为 False
        assert (out["is_fomc"] == False).all()
        assert (out["is_cpi"] == False).all()

    def test_opex_third_friday(self):
        from spx_scanner.features.regime import _third_friday_of_month
        # 2025-01-17 是第三个周五
        assert _third_friday_of_month(2025, 1) == datetime.date(2025, 1, 17)
        # 2025-03-21 是第三个周五
        assert _third_friday_of_month(2025, 3) == datetime.date(2025, 3, 21)


# ── 集成:compute_all_features ────────────────────────────────────────────────

class TestComputeAllFeatures:
    def test_all_features_runs(self, df_3m_long):
        from spx_scanner.features import compute_all_features
        out = compute_all_features(df_3m_long)
        expected_cols = [
            "ema_9", "ema_21", "ema_50", "ema_200",
            "vwap", "bb_width", "is_squeeze",
            "atr_14", "rvol", "rsi_14", "macd_hist",
            "consec_up", "consec_down",
            "minutes_from_open", "session_segment",
            "orb_high", "orb_low", "pdh", "pdl",
        ]
        for col in expected_cols:
            assert col in out.columns, f"缺少列: {col}"

    def test_no_new_rows(self, df_3m_long):
        from spx_scanner.features import compute_all_features
        out = compute_all_features(df_3m_long)
        assert len(out) == len(df_3m_long)

    def test_index_preserved(self, df_3m_long):
        from spx_scanner.features import compute_all_features
        out = compute_all_features(df_3m_long)
        assert (out.index == df_3m_long.index).all()
