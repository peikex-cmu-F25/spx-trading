# SPX 0DTE Scanner — 参数变更日志

> **规则**: 每次修改 `config/params.yaml` 中的任何参数,必须在此文件记录:
> - 修改原因(基于哪个数据集的观察)
> - 修改前/后的值
> - 训练集/验证集上的效果变化
> - 操作人 & 日期

---

## v0.1.0 — 2026-04-24 (初始基线)

**数据范围**: SPY 1min RTH, 2026-04-06 ~ 2026-04-23 (14 个交易日)

**切分**: Train 70% (≈10天) / Val 15% (≈2天) / Test 15% (≈2天)

### 初始参数设定理由

| Pattern | 关键参数 | 初始值 | 设定依据 |
|---|---|---|---|
| `failed_breakout` | `rvol_threshold` | 1.3 | 文献:量能放大 30% 以上为有效信号 |
| `failed_breakout` | `min_touches` | 3 | 少于 3 次触碰阻力位可靠性低 |
| `orb_breakout` | `orb_duration_min` | 30 | 前 30min 为公认 ORB 形成窗口 |
| `orb_breakout` | `rvol_threshold` | 1.3 | 同上 |
| `squeeze_release` | `squeeze_duration_bars` | 5 | 约 15min 压缩才算有效 |
| `squeeze_release` | `rvol_threshold` | 1.5 | 爆发需要更强量能确认 |
| `vwap_rejection` | `prior_bars_one_side` | 5 | 需在 VWAP 一侧运行 ≥15min |
| `vwap_rejection` | `upper_wick_ratio` | 0.5 | 上影线需超过 bar range 的 50% |
| `liquidity_sweep` | `wick_ratio` | 0.6 | 扫除信号需更显著的影线 |
| `liquidity_sweep` | `rvol_threshold` | 1.5 | 扫除通常伴随量能放大 |
| `last_hour_drift` | `trigger_after_minute` | 300 | 14:30 后进入尾盘时段 |
| `last_hour_drift` | `force_exit_minutes_before_close` | 15 | 15:45 强平避免收盘风险 |

### M7 OOS 观察

- **数据量不足**: 14 天数据使多数 pattern 在 Test 集的样本数 < 30, 结论统计意义有限
- **有效 pattern (Test 集 ≥ 10 样本)**:
  - `orb_breakout`: ~20 笔
  - `last_hour_drift`: ~17 笔
- **统计不足 pattern**: `squeeze_release` / `vwap_rejection` / `failed_breakout` / `liquidity_sweep`

### 待调参建议 (基于 Train+Val 观察, 需更多数据验证)

1. **`orb_breakout.rvol_threshold`**: 考虑从 1.3 → 1.5, 可减少约 30% 的信号但 Win Rate 预计提升
2. **`last_hour_drift.trigger_after_minute`**: 考虑从 300 → 330 (15:00后), 尾盘效应在最后 60min 更强
3. **`vwap_rejection.prior_bars_one_side`**: 考虑从 5 → 3, 当前触发数量太少

> ⚠️ 以上调参建议尚未实施,需收集 ≥ 60 个交易日数据后在 Train+Val 上验证才能更新。

---

## 变更记录模板

```
## vX.Y.Z — YYYY-MM-DD

### 修改: {param_path}
- **修改前**: {old_value}
- **修改后**: {new_value}
- **原因**: {reason}
- **数据集**: {train/val/test — 仅允许基于 train+val}
- **效果**: Train Win Rate {old}→{new}, Val Win Rate {old}→{new}
- **操作人**: {name}
```
