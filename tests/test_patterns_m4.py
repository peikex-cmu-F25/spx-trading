"""
tests/test_patterns_m4.py
--------------------------
M4 新增 pattern 单元测试:
  ORBBreakout / SqueezeRelease / VWAPRejection / LiquiditySweep / LastHourDrift
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spx_scanner.patterns.base import Signal
from spx_scanner.patterns.orb_breakout import ORBBreakout
from spx_scanner.patterns.squeeze_release import SqueezeRelease
from spx_scanner.patterns.vwap_rejection import VWAPRejection
from spx_scanner.patterns.liquidity_sweep import LiquiditySweep
from spx_scanner.patterns.last_hour_drift import LastHourDrift
from spx_scanner.patterns.registry import get_all_patterns


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_index(n: int, start: str = "2026-04-07 09:30", freq: str = "3min") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq, tz="US/Eastern")


def _base_df(n: int = 60) -> pd.DataFrame:
    """通用 OHLCV 底板,价格在 500 附近随机游走。"""
    rng = np.random.default_rng(42)
    closes = 500 + np.cumsum(rng.normal(0, 0.1, n))
    opens = closes - rng.uniform(-0.05, 0.05, n)
    highs = np.maximum(opens, closes) + rng.uniform(0.01, 0.1, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.01, 0.1, n)
    volume = rng.integers(10_000, 50_000, n).astype(float)

    idx = _make_index(n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ORBBreakout
# ─────────────────────────────────────────────────────────────────────────────

class TestORBBreakout:
    def _make_df(self, n: int = 80) -> pd.DataFrame:
        df = _base_df(n)
        orb_high = df["close"].iloc[:10].max() + 0.5
        orb_low = df["close"].iloc[:10].min() - 0.5
        df["orb_high"] = orb_high
        df["orb_low"] = orb_low
        df["rvol"] = 1.5
        # minutes_from_open: 每根 bar 3min
        df["minutes_from_open"] = [i * 3 for i in range(n)]
        return df

    def test_instantiate(self):
        pat = ORBBreakout()
        assert pat.name == "orb_breakout"

    def test_missing_columns_raises(self):
        pat = ORBBreakout()
        with pytest.raises(ValueError, match="缺少特征列"):
            pat.detect(pd.DataFrame({"close": [1.0]}))

    def test_detect_returns_list(self):
        pat = ORBBreakout()
        df = self._make_df()
        sigs = pat.detect(df)
        assert isinstance(sigs, list)

    def test_call_signal_on_upside_breakout(self):
        """强制制造一个 close > orb_high 的 call 触发。"""
        pat = ORBBreakout()
        df = self._make_df(80)
        orb_high = df["orb_high"].iloc[0]
        # 找到 minutes_from_open > 30 的第一根 bar,强制让它突破
        for idx in range(len(df)):
            if df["minutes_from_open"].iloc[idx] > 30:
                df.iloc[idx, df.columns.get_loc("close")] = orb_high + 1.0
                df.iloc[idx, df.columns.get_loc("open")] = orb_high + 0.5   # 阳线
                df.iloc[idx, df.columns.get_loc("high")] = orb_high + 1.5
                break
        sigs = pat.detect(df)
        calls = [s for s in sigs if s.direction == "call"]
        assert len(calls) >= 1

    def test_put_signal_on_downside_break(self):
        pat = ORBBreakout()
        df = self._make_df(80)
        orb_low = df["orb_low"].iloc[0]
        for idx in range(len(df)):
            if df["minutes_from_open"].iloc[idx] > 30:
                df.iloc[idx, df.columns.get_loc("close")] = orb_low - 1.0
                df.iloc[idx, df.columns.get_loc("open")] = orb_low - 0.5   # 阴线
                df.iloc[idx, df.columns.get_loc("low")] = orb_low - 1.5
                break
        sigs = pat.detect(df)
        puts = [s for s in sigs if s.direction == "put"]
        assert len(puts) >= 1

    def test_signal_stop_on_correct_side_call(self):
        pat = ORBBreakout()
        df = self._make_df(80)
        orb_high = df["orb_high"].iloc[0]
        for idx in range(len(df)):
            if df["minutes_from_open"].iloc[idx] > 30:
                df.iloc[idx, df.columns.get_loc("close")] = orb_high + 1.0
                df.iloc[idx, df.columns.get_loc("open")] = orb_high + 0.5
                df.iloc[idx, df.columns.get_loc("high")] = orb_high + 1.5
                break
        sigs = [s for s in pat.detect(df) if s.direction == "call"]
        for s in sigs:
            assert s.stop_level < s.entry_price

    def test_explain_returns_string(self):
        pat = ORBBreakout()
        df = self._make_df(80)
        orb_high = df["orb_high"].iloc[0]
        for idx in range(len(df)):
            if df["minutes_from_open"].iloc[idx] > 30:
                df.iloc[idx, df.columns.get_loc("close")] = orb_high + 1.0
                df.iloc[idx, df.columns.get_loc("open")] = orb_high + 0.5
                df.iloc[idx, df.columns.get_loc("high")] = orb_high + 1.5
                break
        sigs = [s for s in pat.detect(df) if s.direction == "call"]
        if sigs:
            assert isinstance(pat.explain(sigs[0]), str)

    def test_no_signal_before_orb_forms(self):
        pat = ORBBreakout()
        df = self._make_df(80)
        orb_high = df["orb_high"].iloc[0]
        # 强制 bar 0 突破,但 minutes_from_open=0 应被过滤
        df.iloc[0, df.columns.get_loc("close")] = orb_high + 2.0
        df.iloc[0, df.columns.get_loc("open")] = orb_high + 1.0
        sigs = [s for s in pat.detect(df)
                if s.timestamp == df.index[0]]
        assert len(sigs) == 0


# ─────────────────────────────────────────────────────────────────────────────
# SqueezeRelease
# ─────────────────────────────────────────────────────────────────────────────

class TestSqueezeRelease:
    def _make_df(self, n: int = 60) -> pd.DataFrame:
        df = _base_df(n)
        df["is_squeeze"] = True
        df["squeeze_duration"] = 8
        df["bb_upper"] = df["close"] + 0.5
        df["bb_lower"] = df["close"] - 0.5
        df["rvol"] = 2.0
        df["atr_14"] = 0.3
        df["atr_ratio"] = 0.5
        df["bb_width_pctile"] = 0.08
        return df

    def test_instantiate(self):
        pat = SqueezeRelease()
        assert pat.name == "squeeze_release"

    def test_missing_columns_raises(self):
        pat = SqueezeRelease()
        with pytest.raises(ValueError, match="缺少特征列"):
            pat.detect(pd.DataFrame({"close": [1.0]}))

    def test_detect_returns_list(self):
        pat = SqueezeRelease()
        df = self._make_df()
        sigs = pat.detect(df)
        assert isinstance(sigs, list)

    def test_call_on_upside_breakout(self):
        pat = SqueezeRelease()
        df = self._make_df(30)
        # bar 10 向上突破 bb_upper
        df.iloc[10, df.columns.get_loc("close")] = df["bb_upper"].iloc[10] + 0.1
        sigs = [s for s in pat.detect(df) if s.direction == "call"]
        assert len(sigs) >= 1

    def test_put_on_downside_breakout(self):
        pat = SqueezeRelease()
        df = self._make_df(30)
        df.iloc[10, df.columns.get_loc("close")] = df["bb_lower"].iloc[10] - 0.1
        sigs = [s for s in pat.detect(df) if s.direction == "put"]
        assert len(sigs) >= 1

    def test_no_signal_when_not_squeeze(self):
        pat = SqueezeRelease()
        df = self._make_df(30)
        df["is_squeeze"] = False
        sigs = pat.detect(df)
        assert len(sigs) == 0

    def test_no_signal_low_rvol(self):
        pat = SqueezeRelease()
        df = self._make_df(30)
        df["rvol"] = 0.8
        df.iloc[10, df.columns.get_loc("close")] = df["bb_upper"].iloc[10] + 0.1
        sigs = pat.detect(df)
        assert len(sigs) == 0

    def test_stop_below_entry_for_call(self):
        pat = SqueezeRelease()
        df = self._make_df(30)
        df.iloc[10, df.columns.get_loc("close")] = df["bb_upper"].iloc[10] + 0.1
        sigs = [s for s in pat.detect(df) if s.direction == "call"]
        for s in sigs:
            assert s.stop_level < s.entry_price

    def test_explain_returns_string(self):
        pat = SqueezeRelease()
        df = self._make_df(30)
        df.iloc[10, df.columns.get_loc("close")] = df["bb_upper"].iloc[10] + 0.1
        sigs = pat.detect(df)
        if sigs:
            assert isinstance(pat.explain(sigs[0]), str)


# ─────────────────────────────────────────────────────────────────────────────
# VWAPRejection
# ─────────────────────────────────────────────────────────────────────────────

class TestVWAPRejection:
    def _make_below_df(self, n: int = 40) -> pd.DataFrame:
        """制造 prior_bars 根 close < vwap 的场景,用于 PUT 测试。"""
        df = _base_df(n)
        vwap_val = 501.0
        df["vwap"] = vwap_val
        # ema_21 下斜:每根 bar 递减,保证 ema21[i] < ema21[i-5]
        df["ema_21"] = [vwap_val - 0.5 - j * 0.02 for j in range(n)]
        df["rvol"] = 1.5
        # 所有 close 在 VWAP 下方
        df["close"] = vwap_val - 1.0
        df["open"] = vwap_val - 1.1
        df["high"] = vwap_val - 0.8
        df["low"] = vwap_val - 1.5
        return df

    def test_instantiate(self):
        pat = VWAPRejection()
        assert pat.name == "vwap_rejection"

    def test_missing_columns_raises(self):
        pat = VWAPRejection()
        with pytest.raises(ValueError, match="缺少特征列"):
            pat.detect(pd.DataFrame({"close": [1.0]}))

    def test_detect_returns_list(self):
        pat = VWAPRejection()
        df = self._make_below_df()
        sigs = pat.detect(df)
        assert isinstance(sigs, list)

    def test_put_signal_wick_rejection(self):
        """prior N 根 close < vwap,当前 bar high >= vwap 但 close < vwap,且上影线 > 0.5。"""
        pat = VWAPRejection()
        df = self._make_below_df(30)
        vwap_val = df["vwap"].iloc[0]
        i = 20
        # 上影线:high 刺穿 VWAP,close 收回
        df.iloc[i, df.columns.get_loc("high")] = vwap_val + 0.3
        df.iloc[i, df.columns.get_loc("close")] = vwap_val - 0.1
        df.iloc[i, df.columns.get_loc("open")] = vwap_val - 0.15
        df.iloc[i, df.columns.get_loc("low")] = vwap_val - 0.2
        sigs = [s for s in pat.detect(df) if s.direction == "put"]
        assert len(sigs) >= 1

    def test_call_signal_wick_rejection(self):
        """prior N 根 close > vwap,当前 bar low <= vwap 但 close > vwap。"""
        pat = VWAPRejection()
        n = 30
        df = _base_df(n)
        vwap_val = 500.0
        df["vwap"] = vwap_val
        # ema_21 上斜:保证 ema21[i] > ema21[i-5]
        df["ema_21"] = [vwap_val + 0.5 + j * 0.02 for j in range(n)]
        df["rvol"] = 1.5
        # 所有 close 在 VWAP 上方
        df["close"] = vwap_val + 1.0
        df["open"] = vwap_val + 0.9
        df["high"] = vwap_val + 1.5
        df["low"] = vwap_val + 0.5
        i = 20
        df.iloc[i, df.columns.get_loc("low")] = vwap_val - 0.3
        df.iloc[i, df.columns.get_loc("close")] = vwap_val + 0.1
        df.iloc[i, df.columns.get_loc("open")] = vwap_val + 0.05
        df.iloc[i, df.columns.get_loc("high")] = vwap_val + 0.2
        sigs = [s for s in pat.detect(df) if s.direction == "call"]
        assert len(sigs) >= 1

    def test_no_signal_low_rvol(self):
        pat = VWAPRejection()
        df = self._make_below_df(30)
        df["rvol"] = 0.5
        sigs = pat.detect(df)
        assert len(sigs) == 0

    def test_stop_above_entry_for_put(self):
        pat = VWAPRejection()
        df = self._make_below_df(30)
        vwap_val = df["vwap"].iloc[0]
        i = 20
        df.iloc[i, df.columns.get_loc("high")] = vwap_val + 0.3
        df.iloc[i, df.columns.get_loc("close")] = vwap_val - 0.1
        df.iloc[i, df.columns.get_loc("open")] = vwap_val - 0.15
        df.iloc[i, df.columns.get_loc("low")] = vwap_val - 0.2
        sigs = [s for s in pat.detect(df) if s.direction == "put"]
        for s in sigs:
            assert s.stop_level > s.entry_price

    def test_explain_returns_string(self):
        pat = VWAPRejection()
        df = self._make_below_df(30)
        vwap_val = df["vwap"].iloc[0]
        i = 20
        df.iloc[i, df.columns.get_loc("high")] = vwap_val + 0.3
        df.iloc[i, df.columns.get_loc("close")] = vwap_val - 0.1
        df.iloc[i, df.columns.get_loc("open")] = vwap_val - 0.15
        df.iloc[i, df.columns.get_loc("low")] = vwap_val - 0.2
        sigs = pat.detect(df)
        if sigs:
            assert isinstance(pat.explain(sigs[0]), str)


# ─────────────────────────────────────────────────────────────────────────────
# LiquiditySweep
# ─────────────────────────────────────────────────────────────────────────────

class TestLiquiditySweep:
    def _make_df(self, n: int = 40) -> pd.DataFrame:
        df = _base_df(n)
        df["rvol"] = 2.0
        df["pdh"] = 502.0
        df["pdl"] = 498.0
        df["orb_high"] = 501.5
        df["orb_low"] = 498.5
        df["prior_hour_high"] = 501.0
        df["prior_hour_low"] = 499.0
        return df

    def test_instantiate(self):
        pat = LiquiditySweep()
        assert pat.name == "liquidity_sweep"

    def test_missing_columns_raises(self):
        pat = LiquiditySweep()
        with pytest.raises(ValueError, match="缺少特征列"):
            pat.detect(pd.DataFrame({"close": [1.0]}))

    def test_detect_returns_list(self):
        pat = LiquiditySweep()
        df = self._make_df()
        sigs = pat.detect(df)
        assert isinstance(sigs, list)

    def test_put_on_high_sweep(self):
        """bar high 刺穿 pdh,close 收回 + 上影线大。"""
        pat = LiquiditySweep()
        df = self._make_df(20)
        pdh = 502.0
        i = 10
        # high > pdh, close < pdh, 大上影线
        df.iloc[i, df.columns.get_loc("high")] = pdh + 0.5
        df.iloc[i, df.columns.get_loc("close")] = pdh - 0.2
        df.iloc[i, df.columns.get_loc("open")] = pdh - 0.3
        df.iloc[i, df.columns.get_loc("low")] = pdh - 0.35
        sigs = [s for s in pat.detect(df) if s.direction == "put"]
        assert len(sigs) >= 1

    def test_call_on_low_sweep(self):
        """bar low 刺穿 pdl,close 收回 + 下影线大。"""
        pat = LiquiditySweep()
        df = self._make_df(20)
        pdl = 498.0
        i = 10
        df.iloc[i, df.columns.get_loc("low")] = pdl - 0.5
        df.iloc[i, df.columns.get_loc("close")] = pdl + 0.2
        df.iloc[i, df.columns.get_loc("open")] = pdl + 0.3
        df.iloc[i, df.columns.get_loc("high")] = pdl + 0.35
        sigs = [s for s in pat.detect(df) if s.direction == "call"]
        assert len(sigs) >= 1

    def test_no_signal_low_rvol(self):
        pat = LiquiditySweep()
        df = self._make_df(20)
        df["rvol"] = 0.5
        sigs = pat.detect(df)
        assert len(sigs) == 0

    def test_put_stop_above_entry(self):
        pat = LiquiditySweep()
        df = self._make_df(20)
        pdh = 502.0
        i = 10
        df.iloc[i, df.columns.get_loc("high")] = pdh + 0.5
        df.iloc[i, df.columns.get_loc("close")] = pdh - 0.2
        df.iloc[i, df.columns.get_loc("open")] = pdh - 0.3
        df.iloc[i, df.columns.get_loc("low")] = pdh - 0.35
        sigs = [s for s in pat.detect(df) if s.direction == "put"]
        for s in sigs:
            assert s.stop_level > s.entry_price

    def test_call_stop_below_entry(self):
        pat = LiquiditySweep()
        df = self._make_df(20)
        pdl = 498.0
        i = 10
        df.iloc[i, df.columns.get_loc("low")] = pdl - 0.5
        df.iloc[i, df.columns.get_loc("close")] = pdl + 0.2
        df.iloc[i, df.columns.get_loc("open")] = pdl + 0.3
        df.iloc[i, df.columns.get_loc("high")] = pdl + 0.35
        sigs = [s for s in pat.detect(df) if s.direction == "call"]
        for s in sigs:
            assert s.stop_level < s.entry_price

    def test_explain_returns_string(self):
        pat = LiquiditySweep()
        df = self._make_df(20)
        pdh = 502.0
        i = 10
        df.iloc[i, df.columns.get_loc("high")] = pdh + 0.5
        df.iloc[i, df.columns.get_loc("close")] = pdh - 0.2
        df.iloc[i, df.columns.get_loc("open")] = pdh - 0.3
        df.iloc[i, df.columns.get_loc("low")] = pdh - 0.35
        sigs = pat.detect(df)
        if sigs:
            assert isinstance(pat.explain(sigs[0]), str)


# ─────────────────────────────────────────────────────────────────────────────
# LastHourDrift
# ─────────────────────────────────────────────────────────────────────────────

class TestLastHourDrift:
    def _make_df(self, n: int = 60) -> pd.DataFrame:
        """尾盘时段(14:30 后)模拟数据。"""
        # 以 14:00 为起点,每根 bar 3min,到约 17:00
        idx = pd.date_range("2026-04-07 14:00", periods=n, freq="3min", tz="US/Eastern")
        rng = np.random.default_rng(7)
        closes = 500 + np.cumsum(rng.normal(0.05, 0.05, n))  # 轻微上涨趋势
        opens = closes - rng.uniform(-0.02, 0.02, n)
        highs = np.maximum(opens, closes) + rng.uniform(0.01, 0.05, n)
        lows = np.minimum(opens, closes) - rng.uniform(0.01, 0.05, n)
        vwap = closes - 0.3  # close > vwap(上涨趋势)
        ema21 = closes - 0.2
        rvol = np.full(n, 1.5)

        # minutes_from_open: 09:30 → 14:00 = 270min
        minutes_from_open = [270 + i * 3 for i in range(n)]
        # minutes_to_close: 16:00 → 14:00 = 120min
        minutes_to_close = [120 - i * 3 for i in range(n)]

        return pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": rng.integers(5_000, 30_000, n).astype(float),
            "vwap": vwap, "ema_21": ema21, "rvol": rvol,
            "minutes_from_open": minutes_from_open,
            "minutes_to_close": minutes_to_close,
        }, index=idx)

    def test_instantiate(self):
        pat = LastHourDrift()
        assert pat.name == "last_hour_drift"

    def test_missing_columns_raises(self):
        pat = LastHourDrift()
        with pytest.raises(ValueError, match="缺少特征列"):
            pat.detect(pd.DataFrame({"close": [1.0]}))

    def test_detect_returns_list(self):
        pat = LastHourDrift()
        df = self._make_df()
        sigs = pat.detect(df)
        assert isinstance(sigs, list)

    def test_call_signal_bullish_trend(self):
        """close > vwap + ema 上斜 + 阳线 → call。"""
        pat = LastHourDrift()
        df = self._make_df(50)
        # 修改 ema_21 明确上斜
        for j in range(len(df)):
            df.iloc[j, df.columns.get_loc("ema_21")] = 499 + j * 0.05
        sigs = [s for s in pat.detect(df) if s.direction == "call"]
        assert len(sigs) >= 1

    def test_no_signal_before_trigger_time(self):
        """minutes_from_open <= trigger_after_minute → 无信号。"""
        pat = LastHourDrift()
        df = self._make_df(50)
        # 把所有 bar 设为开盘 200min 内
        df["minutes_from_open"] = [200 + i for i in range(50)]
        sigs = pat.detect(df)
        assert len(sigs) == 0

    def test_no_signal_near_close(self):
        """minutes_to_close <= force_exit_before → 无信号。"""
        pat = LastHourDrift()
        df = self._make_df(50)
        df["minutes_to_close"] = 5  # 仅剩 5 分钟
        sigs = pat.detect(df)
        assert len(sigs) == 0

    def test_put_signal_bearish_trend(self):
        """close < vwap + ema 下斜 + 阴线 → put。"""
        pat = LastHourDrift()
        n = 50
        df = self._make_df(n)
        # 翻转:close < vwap
        df["vwap"] = df["close"] + 0.5
        df["ema_21"] = df["close"] + 0.3
        # ema 下斜
        for j in range(len(df)):
            df.iloc[j, df.columns.get_loc("ema_21")] = 502 - j * 0.05
        # 让 close < open(阴线)
        df["open"] = df["close"] + 0.1
        sigs = [s for s in pat.detect(df) if s.direction == "put"]
        assert len(sigs) >= 1

    def test_hold_capped_by_close(self):
        """suggested_hold_min 不超过剩余时间。"""
        pat = LastHourDrift()
        df = self._make_df(50)
        for j in range(len(df)):
            df.iloc[j, df.columns.get_loc("ema_21")] = 499 + j * 0.05
        sigs = pat.detect(df)
        for s in sigs:
            ctx_mtc = s.context.get("minutes_to_close", 999)
            assert s.suggested_hold_min <= ctx_mtc

    def test_explain_returns_string(self):
        pat = LastHourDrift()
        df = self._make_df(50)
        for j in range(len(df)):
            df.iloc[j, df.columns.get_loc("ema_21")] = 499 + j * 0.05
        sigs = pat.detect(df)
        if sigs:
            assert isinstance(pat.explain(sigs[0]), str)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_get_all_patterns_returns_list(self):
        pats = get_all_patterns()
        assert isinstance(pats, list)
        assert len(pats) >= 1

    def test_all_patterns_have_name(self):
        for p in get_all_patterns():
            assert isinstance(p.name, str) and len(p.name) > 0

    def test_all_expected_patterns_present(self):
        names = {p.name for p in get_all_patterns()}
        expected = {
            "failed_breakout", "orb_breakout", "squeeze_release",
            "vwap_rejection", "liquidity_sweep", "last_hour_drift",
        }
        assert expected.issubset(names)
