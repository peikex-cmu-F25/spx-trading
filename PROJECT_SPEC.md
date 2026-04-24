# SPX 0DTE 日内爆发点识别系统 — Project Spec

> **核心目标**:构建一个日内 pattern 扫描器,识别 SPY(分析基准) 上高概率方向性爆发点,为 SPX 0DTE 期权的 call/put 入场决策提供信号。
>
> **持有周期**:5–30 分钟
>
> **主时间框架**:3 分钟 K 线(辅助 1min 精细入场、15min 大结构过滤)

---

## 1. 核心假设

市场在大幅方向性移动之前,会留下**可测量的结构性足迹**:
- 波动率压缩(均线粘合、BB 收窄、ATR 下降)
- 关键位反复试探(失败突破、流动性扫除)
- 量能异常(相对量放大、量价背离)
- 时间窗口效应(开盘 ORB、尾盘动量、特定事件日)

系统目标:**把这些足迹量化为可回测、可实时扫描的规则**。

---

## 2. 数据规范

### 2.1 基准与映射

| 用途 | 工具 | 理由 |
|---|---|---|
| **分析基准** | **SPY 1min / 3min OHLCV** | 有真实成交量,pattern 识别依赖量能 |
| **信号映射** | **SPX 0DTE ATM 期权** | SPY × 10 ≈ SPX,结构 pattern 通用 |
| **环境过滤** | **VIX 日线** | Regime 分组 |

**换算关系**:SPY 0.30% 移动 ≈ SPX 约 16–18 点。在 0DTE ATM 上,一般能产生 30–80% 的期权价格变化(取决于 IV 和剩余时间),v1 阶段先用 SPY return 作为代理收益,v2 再加期权定价层。

### 2.2 数据格式(统一 schema)

```
timestamp: pd.Timestamp (tz="US/Eastern")
open:      float
high:      float
low:       float
close:     float
volume:    int
```

### 2.3 时段定义

- **Regular Session(RTH)**:09:30–16:00 ET(主分析时段)
- **Pre-market**:04:00–09:30 ET(保留,用于计算 overnight gap)
- **Post-market**:16:00–20:00 ET(可选,大多数 pattern 不用)

### 2.4 多周期对齐

主周期 **3min** 从 1min 数据重采样:
```python
df_3m = df_1m.resample("3min", origin="09:30:00").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum"
}).dropna()
```
注意 `origin="09:30:00"` 确保 bar 边界对齐开盘。

---

## 3. 系统架构

```
spx_scanner/
├── data_layer/
│   ├── loader.py          # 读 CSV/Parquet,统一时区
│   ├── resampler.py       # 1min → 3min / 15min
│   └── validator.py       # 缺失 bar、异常值检测
│
├── features/
│   ├── structure.py       # 支撑阻力、ORB、PDH/PDL/PDC
│   ├── trend.py           # EMA9/21/50/200、粘合度、VWAP
│   ├── volatility.py      # BB 宽度、ATR、压缩指标、NR7
│   ├── volume.py          # RVOL(按 time-of-day 归一化)、量价一致性
│   ├── momentum.py        # RSI、MACD、连续同向 bar
│   └── regime.py          # VIX 分位、时段标签、星期几、事件日
│
├── patterns/              # 每个文件一个 pattern,接口统一
│   ├── base.py            # Pattern 基类
│   ├── failed_breakout.py
│   ├── squeeze_release.py
│   ├── orb_breakout.py
│   ├── vwap_rejection.py
│   ├── liquidity_sweep.py
│   ├── last_hour_drift.py
│   └── registry.py        # 注册所有 pattern
│
├── scanner/
│   ├── engine.py          # 遍历时间,调用所有 pattern
│   └── signals.py         # Signal 数据结构
│
├── backtest/
│   ├── simulator.py       # 信号 → 模拟持仓 → 退出
│   ├── option_pricer.py   # (v2) BS 近似把 SPY return → SPX option P&L
│   ├── metrics.py         # 胜率、期望、MFE/MAE、夏普
│   └── grouping.py        # 分组统计
│
├── viz/
│   ├── chart.py           # K线+均线+信号标记
│   └── dashboard.py       # Streamlit 面板
│
├── config/
│   └── params.yaml        # 所有阈值(见单独文件)
│
├── tests/                 # 每个 module 配单测
│
└── notebooks/             # 每个里程碑一个 demo notebook
    ├── M1_data_demo.ipynb
    ├── M3_failed_breakout_demo.ipynb
    └── ...
```

---

## 4. 特征定义(v1 基线)

所有阈值在 `config/params.yaml` 中定义,**严禁硬编码**。

### 4.1 结构(`features/structure.py`)

| 特征 | 定义 |
|---|---|
| `resistance_level` | 过去 60 根 3min bar 的 rolling max,被触碰 ≥3 次(价差 ≤0.1%)才认定有效 |
| `support_level` | 镜像 |
| `orb_high`, `orb_low` | 开盘后前 10 根 3min bar(09:30–10:00)的高低 |
| `pdh`, `pdl`, `pdc` | 前一日 high / low / close |
| `touches_to_resistance` | 当前 bar 之前的触碰计数 |

### 4.2 趋势(`features/trend.py`)

| 特征 | 定义 |
|---|---|
| `ema_9`, `ema_21`, `ema_50`, `ema_200` | 对应图里的黄/橙/蓝/红 |
| `ema_compression` | `std([ema_9, ema_21, ema_50]) / close`,< 过去50 bar 的 20 分位 = 粘合 |
| `vwap` | 当日累积 VWAP(每日开盘重置) |
| `vwap_upper_1s`, `vwap_lower_1s` | VWAP ± 1 倍日内标准差 |
| `trend_direction` | `sign(ema_21.diff(5))` + 价格相对 VWAP 位置 |

### 4.3 波动率(`features/volatility.py`)

| 特征 | 定义 |
|---|---|
| `bb_width` | `(bb_upper - bb_lower) / close` |
| `bb_width_pctile` | 过去 N=400 bar(约 5 天 3min)的分位 |
| `is_squeeze` | `bb_width_pctile < 0.20` 且持续 ≥ `K=5` bar |
| `atr_14` | 标准 ATR |
| `atr_ratio` | `atr_14 / atr_14.rolling(400).mean()` |
| `is_nr7` | 当前 bar range 是过去 7 bar 最小 |

### 4.4 量能(`features/volume.py`)

**关键**:成交量必须**按 time-of-day 归一化**,不能直接和隔夜或不同时段比。

```python
# 计算每个 bar 对应的时段平均量(过去 20 天同时段)
df["time_of_day"] = df.index.time
df["vol_tod_mean"] = df.groupby("time_of_day")["volume"].transform(
    lambda x: x.rolling(20, min_periods=5).mean()
)
df["rvol"] = df["volume"] / df["vol_tod_mean"]
```

| 特征 | 定义 |
|---|---|
| `rvol` | 按时段归一化的相对量 |
| `is_high_vol` | `rvol > 1.5` |
| `volume_surge` | 单根 bar rvol > 2.0 |
| `bullish_vol` | 阳线 + 高 rvol |
| `bearish_vol` | 阴线 + 高 rvol |

### 4.5 动量(`features/momentum.py`)

| 特征 | 定义 |
|---|---|
| `rsi_14` | 标准 RSI |
| `macd_hist` | MACD 柱 |
| `macd_hist_flip` | 柱由负转正 / 正转负 |
| `consec_up`, `consec_down` | 连续同向 bar 数 |

### 4.6 环境(`features/regime.py`)

| 特征 | 定义 |
|---|---|
| `vix_level` | 当日 VIX 收盘(或开盘) |
| `vix_regime` | 低<15 / 中 15–20 / 高>20 |
| `minutes_from_open` | 距 09:30 的分钟数 |
| `minutes_to_close` | 距 16:00 的分钟数 |
| `session_segment` | "open" (9:30–10:30) / "midday" (10:30–14:30) / "close" (14:30–16:00) |
| `dow` | 星期几 |
| `is_opex` | 是否月度第三个周五 |
| `is_fomc`, `is_cpi` | 从事件日历 CSV 查(手工维护 `data/events.csv`) |

---

## 5. Pattern 库

### 5.1 统一接口(`patterns/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import pandas as pd

@dataclass
class Signal:
    timestamp: pd.Timestamp
    symbol: str
    pattern: str
    direction: Literal["call", "put"]
    confidence: float          # 0.0 - 1.0
    entry_price: float
    context: dict              # 触发时的关键特征值
    suggested_hold_min: int    # 5 / 10 / 15 / 30
    stop_level: float
    target_level: float | None

class Pattern(ABC):
    name: str
    default_hold_min: int
    
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[Signal]:
        """
        输入:带有所有 feature 列的 DataFrame
        输出:检测到的信号列表
        """
        ...
    
    @abstractmethod
    def explain(self, signal: Signal) -> str:
        """人类可读的触发原因说明,用于复盘"""
        ...
```

### 5.2 Pattern 详细规格

#### 5.2.1 Failed Breakout / Breakdown(优先级 P0)

**你上一张图里那种"突破阻力失败后放量跌破"的 setup。**

**Put 触发条件(Failed Breakout → 做空):**
1. 存在有效 `resistance_level`(过去 60 bar 被触碰 ≥3 次,tolerance 0.1%)
2. 最近 M=10 bar 内至少一次 `high > resistance_level`(尝试突破)
3. 当前 bar `close < resistance_level`(回落到阻力下方)
4. `ema_compression == True` 或 `bb_width_pctile < 0.30`(突破前压缩)
5. 当前 bar 是**阴线 + `rvol > 1.3`**(放量下跌确认)
6. 当前 bar `close < ema_21`(跌破短期均线)

**Call 触发条件(Failed Breakdown)**:镜像。

**Confidence 打分(0–1):**
```
base = 0.5
+ 0.1 if touches_to_resistance >= 4
+ 0.1 if rvol > 2.0
+ 0.1 if bb_width_pctile < 0.15
+ 0.1 if consec_down >= 2 (对 Put)
+ 0.1 if close < ema_50
最终 clip 到 [0, 1]
```

**参数(`patterns.failed_breakout`):**
```yaml
lookback_bars: 60
min_touches: 3
touch_tolerance_pct: 0.001
breakout_window_bars: 10
rvol_threshold: 1.3
squeeze_pctile: 0.30
default_hold_min: 15
stop_buffer_pct: 0.002  # 止损放在阻力位上方 0.2%
```

---

#### 5.2.2 Squeeze Release(优先级 P1)

**波动率极度压缩后单边爆发。**

**触发条件:**
1. `is_squeeze == True` 已持续 ≥ `K=5` bar
2. 当前 bar 出现方向性突破:`close > bb_upper`(做多)或 `close < bb_lower`(做空)
3. `rvol > 1.5`
4. `atr_ratio < 0.7`(全局 ATR 也偏低,压缩是真的)

**方向**:顺突破方向

**Confidence 打分**:压缩越深、量能越大分越高

**参数:**
```yaml
squeeze_duration_bars: 5
bb_width_pctile_max: 0.20
rvol_threshold: 1.5
default_hold_min: 20
```

---

#### 5.2.3 ORB Breakout(优先级 P0)

**开盘前 30 分钟区间被突破。**

**触发条件:**
1. `minutes_from_open > 30`(ORB 已形成)
2. 当前 bar `close > orb_high`(做多)或 `close < orb_low`(做空)
3. 突破 bar `rvol > 1.3`
4. 突破 bar **收在区间外**(非长上/下影 fakeout)

**过滤:**
- ORB 范围过小(< 前 20 日平均 ORB 的 50%)则忽略(可能假信号)
- ORB 范围过大(> 前 20 日平均 ORB 的 200%)则降低 confidence

**Default hold**:15–30 分钟(开盘趋势通常持续)

**参数:**
```yaml
orb_duration_min: 30
rvol_threshold: 1.3
orb_size_min_pctile: 0.20
orb_size_max_pctile: 0.95
default_hold_min: 20
```

---

#### 5.2.4 VWAP Rejection(优先级 P1)

**价格从一侧触 VWAP 后被拒绝。**

**Put 触发(从下往上触 VWAP 被拒):**
1. 前 N=5 bar 价格持续 `< VWAP`(下方运行中)
2. 某根 bar 的 `high >= VWAP` 但 `close < VWAP`(插入但收回)
3. 形成**上影线显著**的 bar(`upper_wick / range > 0.5`)或**看跌吞没**
4. `rvol > 1.2`
5. `ema_21` 斜率为负(趋势确认)

**Call 触发**:镜像(从上往下触 VWAP 被拒)

**参数:**
```yaml
prior_bars_one_side: 5
upper_wick_ratio: 0.5
rvol_threshold: 1.2
default_hold_min: 10
```

---

#### 5.2.5 Liquidity Sweep(优先级 P1)

**扫前高/前低/ORB 高低后迅速反转。**

**Put 触发(扫前高后反转):**
1. 当前 bar `high > max(pdh, orb_high, prior_day_high)` 中任一关键位
2. 当前 bar `close < 被扫除的关键位`(失败穿越)
3. 下一根 bar 或当前 bar `rvol > 1.5`
4. 形成长上影(`upper_wick / range > 0.6`)

**Call 触发**:镜像

**典型场景**:开盘拉高扫前高做空、午后砸盘扫前低做多

**参数:**
```yaml
sweep_levels: ["pdh", "pdl", "orb_high", "orb_low", "prior_hour_high", "prior_hour_low"]
wick_ratio: 0.6
rvol_threshold: 1.5
default_hold_min: 15
```

---

#### 5.2.6 Last Hour Drift(优先级 P2)

**尾盘(15:00 后)顺日内趋势加速。**

**触发条件:**
1. `minutes_to_close <= 60`
2. 日内趋势明确:`close vs vwap` 和 `ema_21 slope` 方向一致
3. 当前 bar 顺势(阳线顺多头 / 阴线顺空头)
4. VWAP 未被反向穿越超过 N=3 bar

**方向**:顺日内趋势

**特殊**:0DTE gamma pin 效应在这个时段最强,建议持有至收盘或 15:45 强平

**参数:**
```yaml
trigger_after_minute: 300  # 14:30 ET
max_counter_trend_bars: 3
default_hold_min: 30
force_exit_minutes_before_close: 15
```

---

## 6. 扫描引擎(`scanner/engine.py`)

```python
class ScannerEngine:
    def __init__(self, patterns: list[Pattern], config: dict):
        self.patterns = patterns
        self.config = config
    
    def scan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入:已计算所有 feature 的 DataFrame
        输出:signals DataFrame,columns = [timestamp, pattern, direction, confidence, ...]
        """
        all_signals = []
        for pattern in self.patterns:
            signals = pattern.detect(df)
            all_signals.extend(signals)
        return pd.DataFrame([s.__dict__ for s in all_signals]).sort_values("timestamp")
    
    def scan_live(self, latest_bar: pd.Series, history: pd.DataFrame):
        """实时模式:只检查最新 bar 是否触发(性能优化)"""
        ...
```

---

## 7. 回测协议(`backtest/simulator.py`)

### 7.1 入场规则
- 信号产生在 bar `t` 收盘时
- **入场价 = bar `t+1` 的 open**(模拟真实延迟和滑点)
- 滑点:加 1 bps(SPY 约 0.05 USD)

### 7.2 退出规则(三种,并行对比)

| 策略 | 描述 |
|---|---|
| **Fixed Time** | 持有 pattern 的 `default_hold_min`,到点平仓 |
| **Target/Stop** | 到 `target_level` 或 `stop_level`,否则按 fixed time 兜底 |
| **Trailing ATR** | 入场后以 1×ATR 为跟踪止损 |

### 7.3 评估指标(`backtest/metrics.py`)

**基础:**
- 交易数 N、胜率、平均盈利、平均亏损、盈亏比、期望值
- MFE(最大有利偏移)、MAE(最大不利偏移)
- 最大连败、最大连胜

**风险调整:**
- 夏普比率(按 trade-level,不是日线)
- Profit Factor = 总盈利 / 总亏损
- Calmar(年化收益 / 最大回撤)

**质量诊断:**
- **置信度校准曲线**:置信度 vs 实际胜率,应单调
- **学习曲线**:按时间切块看胜率稳定性
- **时段热力图**:不同时段胜率

### 7.4 分组分析(`backtest/grouping.py`)

必做的分组(每个 pattern 都要看):
- By `session_segment`(open / midday / close)
- By `vix_regime`(低/中/高)
- By `dow`(周一…周五)
- By `confidence` 分桶(0.5-0.6 / 0.6-0.7 / 0.7-0.8 / 0.8+)
- By `is_opex` / `is_fomc`(事件日差异)

### 7.5 样本外保护(铁律)

- 数据按时间**切三段**:训练 70% / 验证 15% / 测试 15%
- 调参**只在训练+验证**上做
- 测试集**一次性评估**,不允许回头改参数
- 每个 pattern 至少 **N ≥ 30 个测试集样本** 才认为结论有效,少于此数的 pattern 标为"统计不足"

---

## 8. 可视化(`viz/`)

### 8.1 单信号复盘图(`viz/chart.py`)

```python
plot_signal_context(
    df: pd.DataFrame,
    signal: Signal,
    bars_before: int = 30,
    bars_after: int = 30,
) -> matplotlib.figure.Figure
```

图上要有:
- K 线(主图)
- EMA 9/21/50/200(黄/橙/蓝/红,对应你截图)
- VWAP(紫)+ VWAP±1σ 带(浅紫填充)
- 触发时的阻力/支撑水平线
- 入场点(大箭头,方向对应 call/put)
- 出场点(小箭头,颜色按盈亏)
- 成交量子图 + RVOL 线

### 8.2 Streamlit 仪表板(`viz/dashboard.py`)

**页面 1:当日扫描**
- 左:日期选择 + pattern 多选 + 最小置信度滑块
- 中:当日 K 线图 + 所有信号标记
- 右:信号列表 + 点击跳转到复盘图

**页面 2:历史统计**
- Pattern 胜率矩阵(pattern × session_segment)
- 置信度校准曲线
- 月度 / 周度 P&L 曲线
- 最佳/最差 10 个交易的复盘快照

**页面 3:实时信号**(v2)
- 连接实时数据源,最新一根 bar 是否触发

---

## 9. 开发里程碑

### M1:数据层(预估 0.5 天)
**交付:**
- `data_layer/*` 三个 module
- `notebooks/M1_data_demo.ipynb`:读入 SPY 分钟数据 → 画出一天的 3min K 线
- `tests/test_data_layer.py`

**验收:**能正确读取并画出任意一天的 RTH K 线,gap 和 volume 显示正确。

### M2:特征层(预估 1 天)
**交付:**
- `features/*` 六个 module,每个特征有单测
- `notebooks/M2_features_demo.ipynb`:任选一天,画出所有特征的时序

**验收:**每个特征在抽样的 5 天数据上**人工抽检**与 TradingView 等专业平台对比,误差 <1%。

### M3:Failed Breakout 完整闭环(预估 1 天)
**交付:**
- `patterns/failed_breakout.py`
- `scanner/engine.py` 基础版
- `backtest/simulator.py` 基础版
- `viz/chart.py`
- `notebooks/M3_failed_breakout_demo.ipynb`:
  - 在过去 60 天数据上扫描 → 列出所有信号
  - 回测 3 种退出策略 → 打印指标表
  - 随机抽 10 个信号画复盘图

**验收:**
- 至少 30 个历史信号
- 能人工看出"这就是 Failed Breakout"(算法和眼睛一致)
- 回测指标合理(胜率不为 0%、不为 100%)

### M4:扩展 Pattern 库(预估 2 天)
按 P0 → P1 → P2 顺序依次添加:
1. ORB Breakout(P0)
2. Squeeze Release(P1)
3. VWAP Rejection(P1)
4. Liquidity Sweep(P1)
5. Last Hour Drift(P2)

每加一个,单独跑 demo notebook 验证。

### M5:完整回测框架(预估 1 天)
**交付:**
- `backtest/metrics.py` 全套指标
- `backtest/grouping.py` 全套分组
- `notebooks/M5_full_backtest.ipynb`:所有 pattern 综合回测报告

### M6:Streamlit 仪表板(预估 1 天)
**交付:**
- `viz/dashboard.py`
- 一键启动脚本 `run_dashboard.sh`

### M7:调参与 OOS 验证(预估 1 天)
**交付:**
- `notebooks/M7_oos_validation.ipynb`
- 调参记录(哪些参数改过、为什么改、训练/验证/测试集表现对比)

**总工期**:约 7.5 人日

---

## 10. 防过拟合纪律

1. **样本外留白**:数据最后 15% 严格不参与调参
2. **每个 pattern ≥ 30 样本**才算统计有效
3. **置信度校准**:用 `sklearn.calibration.calibration_curve` 画图验证
4. **交易摩擦**:SPY 按 1 bps 滑点 + $1 per-trade 手续费(v2 换 SPX 期权后换成期权滑点模型)
5. **每次改参数记录日志**(`config/changelog.md`),方便追溯过拟合路径

---

## 11. 技术栈

- **核心**:`pandas`、`numpy`、`pandas-ta`(替代 TA-Lib,pip 安装更简单)
- **可视化**:`mplfinance`(K线)、`plotly`(交互)、`streamlit`(仪表板)
- **数据源**:
  - v1:`yfinance`(免费,SPY 1min 近 30 天)或手动下载的 CSV
  - v2:考虑 Polygon.io / ThetaData(付费,更长历史)
- **测试**:`pytest`
- **配置**:`pyyaml`

---

## 12. 未来扩展(v2+)

- **期权定价层**:BS 模型把 SPY return 换算成 SPX 0DTE ATM 期权 P&L
- **GEX / Dealer Positioning**:从 SpotGamma 或自建期权链数据集
- **机器学习层**:用 pattern 输出 + 特征作为 XGBoost 输入,输出综合置信度
- **实时推送**:Telegram / Discord webhook
- **多标的扩展**:QQQ、IWM 同框架复用
