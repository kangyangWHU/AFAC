# src/tools —— 开发期工具

本目录下的脚本**均不参与推理主链路**。`main.py` / `run.sh` 产出提交结果的过程不会
import 这里的任何模块——它们只在开发调参阶段用来评测、诊断和人工审阅。

## 评测（复现官方三指标，需要带 GT 的训练集）

| 脚本 | 用途 |
|---|---|
| `eval_train_full.py` | 全量训练集评测：LONG + TABLE 各 100 张，出 Text / TEDS / ReadOrder / Overall |
| `eval_long_baseline.py` | LONG 回归基线：缓存重放 100 张，逐文档记分，改动前后比对 `out/long_baseline*.json` |
| `eval_grid.py` | 行列估计诊断（纯几何，不调 API）：骨架行列数 vs 真值 |

## 审计与可视化（生成本地 HTML / 图片，供人工翻查）

| 脚本 | 用途 |
|---|---|
| `audit_strict.py` | 严格零容差一致性审计：只统计不修复，暴露每个 tile 的原始解析是否合格 |
| `audit_double_read.py` | 双读审计：本地小模型作第二证人，与 API 缓存结果对照 |
| `dump_audits.py` | 把管线内 `meta['audit']` 事件逐条出图 |
| `dump_issue_tiles.py` | 问题 tile 审计卡片：tile 原图 + 该区域最终重构结果上下拼图 |
| `render_pred_grids.py` | 把提交 CSV 的预测渲染成图片，与原图并排比对 |
| `build_long_compare.py` | LONG 逐文档对照页：左原图 / 右文本，同步滚动 |
| `build_long_diff.py` | LONG 标题改动审阅页：新旧预测的 `#` 层级差异挂徽章 |
| `build_long_table_audit.py` | LONG 文档内表格的并排审计页 |
| `build_long_table_crops.py` | 把 LONG 里的表格单独裁出来，左原图裁片 / 右重构 `<table>` |

所有脚本以模块方式在 `src/` 目录下运行，例如：

```bash
cd src
python -m tools.eval_train_full
```
