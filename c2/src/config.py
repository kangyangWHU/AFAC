# -*- coding: utf-8 -*-
"""全局配置：路径、数据集定位、API 凭据。

所有路径使用相对定位（以本文件位置为锚），方便复现时整体迁移。
"""
import os

# ---- 目录锚点 ----
SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # c2/src
C2_DIR = os.path.dirname(SRC_DIR)                              # c2
DATA_DIR = os.path.join(C2_DIR, "data")

# ---- 训练集（带 GT，用于本地评测与调参）----
# 目录内含 images/（*.jpg）与 mds/（同名 *.md，即 Ground Truth）
TRAIN_LONG_DIR = os.path.join(
    DATA_DIR, "AFAC 训练数据集_extracted", "finixdocbench_huge_long_100")
TRAIN_TABLE_DIR = os.path.join(
    DATA_DIR, "AFAC 训练数据集_extracted", "finixdocbench_huge_table_100")

# ---- A 榜测试集（仅 images/，无 GT）----
A_LONG_DIR = os.path.join(
    DATA_DIR, "AFAC A榜评测数据集(2)_extracted", "finix_huge_long_rest_A")
A_TABLE_DIR = os.path.join(
    DATA_DIR, "AFAC A榜评测数据集(2)_extracted", "finix_huge_table_rest_A")

# ---- 输出目录 ----
OUT_DIR = os.path.join(C2_DIR, "out")
CACHE_DIR = os.path.join(C2_DIR, "cache")          # API 结果缓存，避免重复调用
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---- FinixDoc-VL API 凭据（见 doc/API.txt）----
API_URL = "https://finixdocapi.alipay.com/api/finix_doc/call_with_file"
API_KEY = "F935A5503983FB19F26FA3F00A94EBF9"           # 比赛统一固定 apiKey
API_USER_IDS = [                                       # 5 个白名单 userId，可轮询负载均衡
    "finixA1001", "finixB2002", "finixC3003", "finixD4004", "finixE5005",
]

# ---- 二值化阈值（灰度 < 阈值 = 墨）：分级可调，供 Stage I 裁剪几何用 ----
# 数据/正文是浓墨(灰度<128)，框线/标题常是淡灰或彩色(灰度130~180)——分严/松两档区分。
BIN_INK = 128        # 严：实墨(浓数据/正文)，排除淡灰/彩色线与淡标题
BIN_FAINT = 180      # 松：含淡灰/彩色框线与淡标题(灰度130~180)，如 f64061da 首标题灰度177
BIN_LINE = 170       # 并排/横线检测(_panel_seams / _panel_seam_xs)

# ---- 图分类阈值（实测：LONG 长宽比 20~128，TABLE 1.4）----
LONG_ASPECT_MIN = 5.0        # 长宽比 ≥ 5 判为 LONG（面条图）

# ---- 并发：官方上限 16，试探 32 看吞吐 ----
# 即便撞限流，返回 HTTP 错误/错误信封 → 重试退避、**不写缓存**（不污染），最坏只是变慢。
MAX_CONCURRENCY = 16

# ---- 评测：表格块识别 ----
# GT 中表格以 HTML <table> 标签出现；据此把文档拆成 文本块 / 表格块
TABLE_OPEN_RE = r"<table[^>]*>"
TABLE_CLOSE_RE = r"</table>"
