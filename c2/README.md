# AFAC 赛题二 · 多模态超大文档解析

超长/超大金融文档截图 → 高保真 Markdown。本工程为 B 榜提交方案的完整源码与复现说明。

---

## 一、快速开始

```bash
# 1. 环境(conda,Python 3.11)
bash init_env.sh                 # 或: pip install -r requirements.txt

# 2. 一键跑出提交
bash run.sh                                  # 数据集在 c2/data/AFACB榜评测数据集
bash run.sh /path/to/AFACB榜评测数据集         # 数据集在别处
```

产出 `out/submission.csv`，两列 `file_name, ground_truth`，共 100 行，UTF-8、全字段加引号、按 `file_name` 排序。

`run.sh` 期望数据根目录下是官方原始结构：

```
<数据根>/finix_huge_long_rest_B/images/*.jpg     # 50 张 面条图(LONG)
<数据根>/finix_huge_table_rest_B/images/*.jpg    # 50 张 大表图(TABLE)
```

跑 A 榜把 `run.sh` 里的 `LONG_SUB` / `TABLE_SUB` 改成 `..._rest_A` 即可。也可直接调入口：

```bash
cd src
python main.py --long_dir <LONG>/images --table_dir <TABLE>/images --out ../out/submission.csv
python main.py --long_dir <LONG>/images --out ../out/long_only.csv --limit 3   # 冒烟测试
```

`--procs` 图级并行进程数（默认 6），`--timeout` 单图超时（默认 240s），`--target_h` LONG 切条目标高度（默认 5000px）。

**类别不靠猜**：`long` / `table` 由输入目录给定——官方数据集本就按两类分目录发布，用长宽比自动分类只会平白引入一类错误。

---

## 二、环境与硬件

| 项 | 要求 |
|---|---|
| Python | 3.11（3.9+ 应可用，未逐版本验证） |
| 操作系统 | Linux（开发机 Ubuntu 6.8 内核）；仅用跨平台库，macOS 应可运行 |
| GPU | **不需要**。全流程 CPU + 远端 API |
| CPU | 建议 ≥ 8 核。开发机 24 核；几何计算是 CPU-bound，核多则快 |
| 内存 | 建议 ≥ 16GB。单页最大 3.8 亿像素，`Image.MAX_IMAGE_PIXELS` 已解除保护 |
| 网络 | 需可访问 `finixdocapi.alipay.com` |
| 磁盘 | ≥ 2GB（API 结果缓存 `cache/`、上采样缓存 `cache_up/`、格级 OCR 缓存 `rec_cache.sqlite`） |

依赖见 `requirements.txt`。`run.sh` 启动前会做一次依赖自检——缺包立刻报错，不会跑到一半才 `ImportError`（`rapidocr` 是惰性导入，只在残差修复分支才加载，最容易漏装）。

### 运行时长

实测（24 核 / 32 并发 / 冷缓存）：TABLE 半边 50 张约 10 分钟，LONG 半边更快，**整批 100 张约 20–30 分钟**，远低于 3h 上限。其中 LONG 的几何定级步骤冷跑约 5s/张（本地逐行 OCR），按图缓存到 `cache_geo/` 后近乎为零，分进程后对总时长影响在一分钟以内。若 API 侧限流，客户端会退避重试，耗时相应拉长；`config.py` 里 `API_TIMEOUT=40 / API_RETRIES=3` 已按"快速放弃 + 靠缓存重跑补失败"的策略调过——重跑时已成功的 tile 直接命中缓存，等于天然的末尾补扫。

---

## 三、合规声明

- **大模型**：全工程唯一的 VLM 接口是主办方提供的 FinixDoc-VL API，调用点集中在 `src/common/api_client.py`（`requests.post` 两处），无任何其它第三方模型 API、SDK 或网络地址。
- **Prompt**：该 API 的请求体只有 `userId / apiKey / fileName + 图片文件`，不接受文本 prompt 参数，因此本方案没有 prompt 配置文件——引导模型的工作全在图像侧（切在哪、放大多少、空白块跳过不送）。
- **本地小模型**：`PP-OCRv6_rec_small.onnx`，**5.27M 参数**（< 10M 上限），CPU + onnxruntime，用于 TABLE 残差格级重读与 LONG 几何标题定级。同引擎载入的 det/cls 模型调用时均置 `use_det=False / use_cls=False` 不参与推理，三者合计也仅 7.72M。模型随 `rapidocr` wheel 安装，运行期不联网。
- **无硬编码**：不含针对特定测试图片的固定输出、白名单或 uuid 分支；所有阈值都是几何/统计维度的通用判据，集中在 `common/config.py` 与 `table/tiles.py`。

---

## 四、目录结构

```
c2/
├── run.sh                    一键端到端脚本(唯一推荐入口)
├── init_env.sh               conda 环境初始化
├── requirements.txt          依赖清单
├── src/
│   ├── main.py               入口:图片目录 → submission.csv(两类同批处理、结果合并、排序写出)
│   ├── common/
│   │   ├── config.py         全局配置:路径、API 凭据、并发/重试、二值化阈值
│   │   ├── api_client.py     FinixDoc-VL 客户端:multipart 上传、userId 轮询、重试退避、
│   │   │                     内容哈希磁盘缓存、错误信封/复读退化识别
│   │   ├── preprocess.py     轻量预处理(仅保证 RGB;重增强实测不涨分,不入默认链路)
│   │   └── imcache.py        纯几何计算的磁盘缓存(几何分表 CPU-bound,重跑时避免重算)
│   ├── long/                 LONG(面条图)流水线
│   │   ├── run_long.py         端到端 runner(run_smart)
│   │   ├── slicer_long.py      1D 水平投影找行间空白带下刀,不把一行字劈成两半
│   │   ├── stitch_long.py      接缝拼接:跨条重复行去重、被切断的表格跨条重连
│   │   ├── heading_norm.py     标题层级校正:栈模型 + 编号序列 + 目录/封面题名处理
│   │   ├── geom_heading.py     几何标题定级:回原图量标题字号/淡横线,做全局层级裁决
│   │   └── table_fix.py        长文内表格结构自愈:满宽横幅行 colspan 归一
│   ├── table/                TABLE(大表图)流水线
│   │   ├── run_table.py        入口 parse_table:三段式 crop → ocr → merge
│   │   ├── crop.py             Stage I 纯几何裁剪:剥表外页眉/页脚/水印、切分并排子表
│   │   ├── geom.py             共享几何原语:投影分段、框线检测、并排缝、表外文字剥离
│   │   ├── slicer_table.py     在网格线处切分,预判空白 tile 直接跳过(不送模型=不幻觉)
│   │   ├── tiles.py            tile 公共层:尺寸/上采样策略常数、图像准备、API 并发调用
│   │   ├── grid_ocr.py         Stage II 骨架 OCR:墨迹几何做尺子,零容差恒等判定,
│   │   │                       不合格 tile 触发唯一修复动作
│   │   ├── cell_ocr.py         残差修复:不一致 tile 回落 PP-OCRv6s 逐格重读(带 sqlite 缓存)
│   │   └── stitch_single.py    Stage III 2D 重组:tile 输出 → 完整 <table>
│   ├── metrics/              本地评测(复现官方三指标)
│   │   ├── evaluate.py         Overall = [(1−TextEdit)×100 + TableTEDS + (1−ReadOrder)×100]/3
│   │   └── teds.py             TEDS 树编辑距离相似度
│   └── tools/                开发期审计/评测工具,**不在推理主链路上**(见 tools/README.md)
├── plan.md                   方案演进记录
└── out/                      运行产出(提交 CSV、日志、评测结果) —— 未纳入版本管理
```

运行期产物（`cache/`、`cache_up/`、`rec_cache.sqlite`、`out/`、`data/`）均已 gitignore，不随代码提交。

---

## 五、方法概要

两类图的痛点完全不同，因此走两条独立流水线。

### LONG（面条图，长宽比 > 30）

单图高达数万像素，整图送模型必然 OOM / 长上下文失效。

1. **切**——水平投影求每行墨量，在**行间空白带**下刀（不是等高裸切）。切点落在字行之间，接缝处就不会出现半行残字，这是 ReadOrder 与 TextEdit 的主要失分源。
2. **读**——各条带并发送 API，按内容哈希缓存。
3. **拼**——`stitch_long` 处理接缝：重叠区重复行去重；长表格中部没有空白带、切点必然落在表内，上条收口成 `</table>`、下条重开 `<table>`，此处做跨条表格重连。
4. **正**——`heading_norm` 修标题层级。模型按条带识别，`#` 是局部判断，拼成整篇后同一编号序列常在条带边界被重置。用栈模型 + 编号相对递减 + 序列历史纠漂移；封面标题提升 L1，目录项的伪标题降级。`table_fix` 再把表内满宽横幅行的 `colspan` 归一（依训练集 GT 主流写法）。
5. **定级**——`geom_heading.correct()` 是 LONG 路的最后一步，也是唯一回到**原图**做全局裁决的一步。前面所有层级判断都基于文本，而文本里"这行是几级标题"本质上是排版信息；这一步回原图逐行量标题的**字号**与其上下的**淡横线**，据此重定 `#` 层级。它必须在拼接与文本定级之后：只有整篇拼好了，字号才有全局可比性。逐行识别用本地 PP-OCRv6s，结果按图片名缓存到 `cache_geo/`。

### TABLE（大表图，单页可达 2 亿像素）

密集网格表，模型读整表会跳行/复读——自回归解码靠自己的输出史追踪表内位置，重复内容摧毁这种追踪。

1. **Stage I 裁剪**（`crop.py`，纯几何不调模型）——剥离表外 furniture（页眉/页脚/水印/页码，判据是"矮 + 窄 + 非最大连通块"），再按横线断裂与列模式周期切分并排子表。竖线默认是列分隔而非分表信号。
2. **Stage II 骨架 OCR**（`grid_ocr.py`）——严格在网格线处切 tile，小字 tile 先上采样；墨量极低的空白 tile **直接跳过不送模型**（送了就是幻觉）。核心是"一把尺、一个判定、一个修复"：墨迹几何（`cell_ink ∪ cell_gray`）给出每个 tile 的期望行数与每行内容宽 → 零容差恒等判定（非空行数与每行有效宽必须完全相符，禁众数、禁 ±1 容差）→ 不合格才触发修复。
3. **残差修复**（`cell_ocr.py`）——诊断仍不一致的 tile 回落本地 PP-OCRv6s **逐格重读**。行列位置由几何骨架给定，模型只回答"这格里是什么"，位置控制权收回代码，**计数类错误（爆行/漏行/行跳读）结构性不可能发生**。单调重复区（如 `3000` × N）恰恰是 API 最容易失稳、逐格识别最容易的地方。本地模型确定性输出，重放天然稳定。
4. **Stage III 2D 重组**（`stitch_single.py`）——tile 结果按行列装配成完整 `<table>`，做列校准与 colspan 审计。

---

## 六、缓存与可复现性

所有 API 结果按**图像内容哈希**落盘缓存（`cache/`、上采样 tile 分离存 `cache_up/`），本地格级 OCR 结果存 `rec_cache.sqlite`，几何定级的逐行 OCR 结果存 `cache_geo/`。作用有三：省额度、重跑失败时只补失败项、以及让同一份输入的重放结果稳定。

需要说明的两点：

- **缓存只存在于本地，不随代码提交**（体积近 400MB，且属运行期产物）。在干净环境首次运行时缓存为空，会真实调用 API 建立缓存。
- **错误响应不入缓存**。服务端偶发吐 HTML 错误页 / 错误信封 / 复读退化（同一行吐上百遍），这些一律判失败并重试，绝不写缓存污染后续重放（见 `api_client.py` 的 `_looks_like_error` 与 `DEGEN_RETRIES`）。

FinixDoc-VL 是生成式模型，同一张图两次调用的输出未必逐字相同，因此"逐字节复现"在原理上依赖服务端行为。本工程的设计把这种不确定性压到最小：几何决策（切哪、跳哪、行列骨架）全部由代码确定性给出，模型只负责识别内容；不合格的 tile 由本地确定性模型兜底。

### 与 B 榜提交结果的一致性（实测）

用本地缓存做全量离线重放（不调用 API），本工程产出与实际提交的 B 榜结果逐字比对：

| 半边 | 逐字一致 |
|---|---|
| LONG 50 张 | **50 / 50** |
| TABLE 50 张 | **50 / 50** |
| **合计** | **100 / 100** |

即 `bash run.sh` 在相同 API 返回下**完全复现**提交结果。流水线本身是确定性的：同一输入连续多次运行输出完全一致，几何决策与本地识别均不含随机性。
