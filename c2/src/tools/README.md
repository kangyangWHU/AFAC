# src/tools —— 开发期工具

本目录下的脚本**均不参与推理主链路**。`main.py` / `run.sh` 产出提交结果的过程不会
import 这里的任何模块——它们只在开发调参阶段用来评测和人工审阅。

以模块方式在 `src/` 目录下运行，例如 `cd src && python -m tools.eval_train_full`。

## 评测（出数字，需带 GT 的训练集）

| 脚本 | 用途 |
|---|---|
| `eval_train_full.py` | 全量训练集评测：LONG + TABLE 各 100 张，出 Text / TEDS / ReadOrder / Overall |
| `eval_long_baseline.py` | LONG 回归基线：缓存重放 100 张，逐文档记分，改动前后比对 `out/long_baseline*.json` |

两者都从 `main.py` 取 `process_image` 进入，量的是与提交完全一致的完整链路。

## 审阅（出图，给人翻）

| 脚本 | 用途 |
|---|---|
| `eval_grid.py` | TABLE 行列骨架诊断（纯几何不调 API）：骨架叠回原图——黄框 seg、绿线行、红线列、蓝线 tile 切分带；有 GT 时对比行列数准确度 |
| `render_pred_grids.py` | 把提交 CSV 的预测重构渲染成图片，与原图并排比对 |
| `build_long_compare.py` | LONG 逐文档对照页：左原图 / 右文本，按比例同步滚动 |
