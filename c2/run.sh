#!/usr/bin/env bash
# AFAC 赛题二 —— 一键端到端复现脚本
#
# 用法:
#   bash run.sh                          # 用默认数据根 c2/data,跑 B 榜测试集
#   bash run.sh <数据集根目录>            # 数据集在别处时指过来
#   OUT=xxx.csv bash run.sh              # 自定义输出路径
#
# 数据根目录下需存在(官方发布的原始目录结构):
#   finix_huge_long_rest_B/images/*.jpg     50 张 面条图
#   finix_huge_table_rest_B/images/*.jpg    50 张 大表图
# 若跑 A 榜,把下面 LONG_SUB/TABLE_SUB 换成 finix_huge_long_rest_A / ..._table_rest_A。
#
# 产出: out/submission.csv (file_name, ground_truth),行按 file_name 排序,共 100 行。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-${AFAC_C2_DATA:-$HERE/data/AFACB榜评测数据集}}"
LONG_SUB="finix_huge_long_rest_B"
TABLE_SUB="finix_huge_table_rest_B"
OUT="${OUT:-$HERE/out/submission.csv}"

LONG_DIR="$DATA_ROOT/$LONG_SUB/images"
TABLE_DIR="$DATA_ROOT/$TABLE_SUB/images"

for d in "$LONG_DIR" "$TABLE_DIR"; do
  [ -d "$d" ] || { echo "[run.sh] 找不到图片目录: $d" >&2; exit 1; }
done

# 依赖自检:缺包直接报错,别跑到一半才 ImportError(rapidocr 是惰性导入的重灾区)
python - <<'PY'
import importlib, sys
missing = [m for m in ("numpy", "PIL", "cv2", "requests", "rapidfuzz",
                       "apted", "lxml", "bs4", "rapidocr", "onnxruntime")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("[run.sh] 缺少依赖: %s —— 先 pip install -r requirements.txt"
             % ", ".join(missing))
PY

echo "[run.sh] LONG : $LONG_DIR"
echo "[run.sh] TABLE: $TABLE_DIR"
echo "[run.sh] OUT  : $OUT"

cd "$HERE/src"
exec python main.py --long_dir "$LONG_DIR" --table_dir "$TABLE_DIR" --out "$OUT"
