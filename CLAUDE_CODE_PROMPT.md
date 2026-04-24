# Claude Code 启动指令 — SPX 0DTE Scanner

> 复制下面的内容到 Claude Code,作为项目的第一条指令。
> 确保 `PROJECT_SPEC.md` 和 `params.yaml` 已放在项目根目录。

---

## 启动 Prompt(复制下方全部内容)

```
你好。这是一个 SPX 0DTE 期权日内 pattern 扫描系统的项目。

请先通读 PROJECT_SPEC.md(完整 spec)和 params.yaml(所有阈值)。
读完后用你自己的话给我总结:
1. 这个项目的核心目标是什么?
2. 系统的 7 个开发里程碑各自交付什么?
3. 有哪些地方你看了 spec 仍然不确定,需要我澄清?

确认对齐后,开始 M1(数据层)的实现。

开发纪律(必须严格遵守):

【代码组织】
- 严格按 spec §3 的目录结构,不要自作主张合并或拆分 module
- 每个 module 的公开函数都要有 type hint 和 docstring
- 所有阈值从 config/params.yaml 读取,严禁硬编码

【测试】
- 每个 module 对应 tests/test_<module>.py
- pytest 运行无红灯才能进入下一个里程碑
- 关键 feature(VWAP、RVOL、EMA 等)要和 TradingView/业界实现对比过

【交付物】
- 每个里程碑结束时产出一个 notebook,路径 notebooks/M<N>_xxx_demo.ipynb
- notebook 要能独立运行、结果可视化、有说明文字

【依赖】
- 用 pandas / numpy / pandas-ta(不用 TA-Lib,安装麻烦)
- K 线画图用 mplfinance,交互图用 plotly
- 仪表板用 streamlit

【推进方式】
- 完成每个里程碑后暂停,给我看 demo notebook 的关键输出截图/描述
- 我确认后再进入下一步
- 不要一口气写完所有代码,每一步都要有验证节点

【数据】
- 我会把 SPY 1min 数据放到 data/spy_1min.parquet
- 如果我还没提供数据,你可以先用 yfinance 拿近 30 天 SPY 1min 数据写 loader demo

【防过拟合底线】
- 最后 15% 时间段数据是测试集,调参阶段严禁接触
- 每个 pattern 至少 30 个样本才算结论有效
- 所有参数调整要在 config/changelog.md 里记录(改了什么、为什么、训练/验证表现)

现在开始,从读 spec + 提澄清问题开始。
```

---

## 你要做的准备工作

### 1. 创建项目目录
```bash
mkdir spx_scanner && cd spx_scanner
# 把 PROJECT_SPEC.md 和 params.yaml 放进来
mkdir -p data config notebooks tests
mv params.yaml config/
```

### 2. 准备数据

**最简单的起步**:让 Claude Code 用 `yfinance` 抓近 30 天 SPY 1min 数据(免费但只有 30 天):
```python
import yfinance as yf
df = yf.download("SPY", period="30d", interval="1m")
df.to_parquet("data/spy_1min.parquet")
```

**更好的选择**(如果你愿意花点钱拿长历史):
- **Polygon.io**:$29/月,SPY 1min 历史可拿 5 年+
- **ThetaData**:按请求量计费,数据质量很高
- **AlgoSeek**:较贵但专业,有 SPX 期权数据

**免费但需手动**:CBOE / Yahoo 手动下载 CSV

### 3. 事件日历

创建 `data/events.csv`(可手工维护):
```csv
date,event_type,notes
2026-01-29,FOMC,Rate decision
2026-02-12,CPI,January CPI release
2026-01-17,OPEX,Monthly options expiration
...
```

## 推进节奏建议

| 时间 | 动作 |
|---|---|
| Day 1 上午 | Claude Code 读 spec + 澄清 + 搭 M1 数据层 |
| Day 1 下午 | M2 特征层(前半) |
| Day 2 | M2 完成 + M3 Failed Breakout 闭环 |
| Day 3–4 | M4 扩展其他 5 个 pattern |
| Day 5 | M5 完整回测 |
| Day 6 | M6 仪表板 |
| Day 7 | M7 OOS 验证 + 调参 |

---

## 回到这个对话的时机

以下几种情况**回来找我**(普通 Claude),因为涉及策略思考而不是纯工程:

1. **M3 Failed Breakout 跑完**:把回测指标和几张典型信号图拿来,我帮你判断 pattern 定义是否需要调整
2. **M5 完整回测报告**:一起分析哪些 pattern 真的有 edge、哪些是噪声
3. **M7 发现过拟合迹象**:训练集和测试集表现差距大时,一起思考调整方向
4. **想加新 pattern / 改特征定义**:先和我讨论定义,再让 Claude Code 实现

Claude Code 负责**把想法变代码**,我负责和你一起**想想法、看结果、做决策**。这样分工最高效。
