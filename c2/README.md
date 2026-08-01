# AFAC 赛题二 · 多模态超大文档解析

超长/超大金融文档截图 → 高保真 Markdown。B 榜提交方案的完整源码与复现说明。

## 一、运行

```bash
bash init_env.sh          # 建 conda 环境 AFAC (Python 3.11) 并装依赖
bash run.sh               # 数据集在 c2/data/AFACB榜评测数据集
bash run.sh <数据集根目录>  # 数据集在别处
```

产出 `out/submission.csv`：两列 `file_name, ground_truth`，100 行，UTF-8、全字段加引号、按 `file_name` 排序。

数据根目录下需是官方原始结构 `finix_huge_long_rest_B/images/` 与 `finix_huge_table_rest_B/images/`（各 50 张）。跑 A 榜把 `run.sh` 里的 `LONG_SUB` / `TABLE_SUB` 改成 `..._rest_A`。

`long` / `table` 由输入目录给定，不靠长宽比猜——官方数据集本就分两类发布，自动分类只会平白引入一类错误。

## 二、环境与硬件

不需要 GPU，全流程 CPU + 远端 API。Python 3.11 / Linux；建议 ≥ 8 核、≥ 16GB 内存（单页最大 3.8 亿像素）、≥ 2GB 磁盘（缓存）；需可访问 `finixdocapi.alipay.com`。依赖见 `requirements.txt`，`run.sh` 启动前会自检。

**运行时长**：实测 24 核 / 32 并发 / 冷缓存，整批 100 张约 30 分钟（B 榜实测两个半边各约 10 分钟，本地小模型的残差重读与几何校准各加一点），远低于 3h 上限。API 限流时客户端退避重试；`API_TIMEOUT=40 / API_RETRIES=3` 按"快速放弃 + 靠缓存重跑补失败"调过——重跑时已成功的 tile 直接命中缓存，等于天然的末尾补扫。

## 三、合规声明

- **大模型**：全工程唯一的 VLM 接口是主办方提供的 FinixDoc-VL API，调用点集中在 `src/common/api_client.py`（`requests.post` 两处），无任何其它第三方模型 API、SDK 或网络地址。
- **Prompt**：该 API 的请求体只有 `userId / apiKey / fileName + 图片文件`，不接受文本 prompt 参数，因此本方案没有 prompt 配置文件——引导模型的工作全在图像侧（切在哪、放大多少、空白块跳过不送）。
- **本地小模型**：`PP-OCRv6_rec_small.onnx`，**5.27M 参数**（< 10M 上限），CPU + onnxruntime，用于 TABLE 残差格级重读与 LONG 几何标题校准。同引擎载入的 det/cls 模型调用时均置 `use_det=False / use_cls=False` 不参与推理，三者合计也仅 7.72M。模型随 `rapidocr` wheel 安装，运行期不联网。
- **无硬编码**：不含针对特定测试图片的固定输出、白名单或 uuid 分支；所有阈值都是几何/统计维度的通用判据，集中在 `common/config.py` 与 `table/tiles.py`。

## 四、目录结构

```
c2/
├── run.sh                    一键端到端脚本(唯一推荐入口)
├── init_env.sh               conda 环境初始化
├── requirements.txt          依赖清单
├── report.md                 方法说明文档
├── src/
│   ├── main.py               入口:图片目录 → submission.csv
│   ├── common/
│   │   ├── config.py         全局配置:路径、API 凭据、并发/重试、二值化阈值
│   │   ├── api_client.py     FinixDoc-VL 客户端:上传、userId 轮询、重试退避、
│   │   │                     内容哈希缓存、错误信封/复读退化识别
│   │   ├── preprocess.py     轻量预处理(仅保证 RGB;重增强实测不涨分)
│   │   └── imcache.py        纯几何计算的磁盘缓存
│   ├── long/                 LONG(面条图)流水线
│   │   ├── run_long.py         端到端 runner(run_smart)
│   │   ├── slicer_long.py      水平投影在行间空白处切分,不切断文字行
│   │   ├── stitch_long.py      接缝拼接:跨条重复行去重、断表跨条重连
│   │   ├── heading_norm.py     标题层级校正:栈模型 + 编号序列 + 目录/封面题名
│   │   ├── geom_heading.py     几何校准:回原图量字号/淡横线,校正标题层级
│   │   └── table_fix.py        长文内表格自愈:满宽横幅行 colspan 归一
│   ├── table/                TABLE(大表图)流水线
│   │   ├── run_table.py        入口 parse_table:crop → ocr → merge
│   │   ├── crop.py             Stage I 纯几何裁剪:剥页眉页脚水印、切子表、判块类型
│   │   ├── geom.py             共享几何原语:行列边界估计、框线检测、并排中缝
│   │   ├── slicer_table.py     在网格线处切分,空白 tile 预判跳过
│   │   ├── tiles.py            tile 公共层:尺寸/上采样策略、并发调用
│   │   ├── grid_ocr.py         Stage II 骨架 OCR:墨迹几何做尺子,零容差判定
│   │   ├── cell_ocr.py         残差修复:不一致 tile 回落 PP-OCRv6s 逐格重读
│   │   └── stitch_single.py    Stage III 2D 重组 → 完整 <table>(含回退路重组)
│   ├── metrics/              本地评测(复现官方三指标)
│   └── tools/                开发期审计/评测工具(见 tools/README.md)
└── out/                      运行产出 —— 未纳入版本管理
```

运行期产物（`cache/`、`cache_up/`、`cache_geo/`、`rec_cache.sqlite`、`out/`、`data/`）均已 gitignore，只存在于本地。

## 五、方法概要

两类图的痛点不同，走两条独立流水线。详细方法见 `report.md`。

### LONG（面条图，长宽比 > 30）

单图高达数万像素，整图送模型必然长上下文失效；宽度锁定在 1500 px、恰落在模型有效分辨率内，因此只需纵向切分。

1. **自适应切分**——水平投影求每行墨量，切点落在行间空白处，接缝不出现半行残字（ReadOrder 与 TextEdit 的主要失分源）；条带内检出表格则避让，但超过目标条高的大表不避让。
2. **并发识别**——各条带并发送 API，按内容哈希缓存。
3. **接缝拼接**——重叠区去重；长表格中部没有行间空白、切点必落在表内，上条被闭合为 `</table>`、下条重开 `<table>`，此处做跨条表格重连。
4. **文本定级**——`heading_norm` 用栈模型按编号连续性定层级，遇此前出现过的编号序列采用其历史层级以免漂移；封面标题合并提升、目录列表项转标题；`table_fix` 归一表内满宽横幅行的 `colspan`。
5. **几何校准**——`geom_heading` 是最后一步，也是唯一回原图的一步：在整篇上重定级，主逻辑仍是编号格式关系，图像证据只裁定一级标题（淡横线上方提升为 H1，无编号中段标题按字号保留或删除标记）。必须放在拼接之后——整篇拼好，字号才有全局可比性。逐行识别用本地 PP-OCRv6s，按图缓存到 `cache_geo/`。

### TABLE（大表图，单页可达 2 亿像素）

密集网格表，模型读整表会跳行/复读——自回归解码靠自己的输出史追踪表内位置，重复内容摧毁这种追踪。

1. **Stage I 几何裁剪**（纯几何，不调模型）——剥离表外页眉/页脚/水印/页码（判据：矮 + 窄 + 非最大连通块）；再切子表，左右并排看横线是否在某个 x 处断裂，上下堆叠看缝高与框线连续性（竖线默认是列分隔，不作分表信号）；最后判定块类型，疑似标题的块留到 Stage II 由模型复核真伪。
2. **Stage II 骨架 OCR**——先估行列骨架（有框表框线即边界，无框表用单元格间的空白缝），严格在网格线处切 tile 使每块含整数行、整数列，密集小字先上采样，墨量极低的空白 tile 直接跳过不送模型（送了就是幻觉）。核心是一把尺：墨迹几何给出每格"应有内容 / 应空"的期望，模型输出须逐格恒等匹配（禁众数、禁 ±1 容差）。
3. **确定性重读**——判定不通过的 tile 整块回退本地 PP-OCRv6s 逐格识别。行列位置由骨架给定，模型只回答"这格是什么"，位置控制权收回代码，**计数类错误（爆行/漏行/跳读）结构性不可能发生**。单调重复区（`3000` × N）恰是 API 最易失稳、逐格识别最容易的地方。
4. **Stage III 2D 重组**——各 tile 按全局行列坐标合并为完整 `<table>`，并做表头重建（列号行锚点 + 跨列 `colspan` 合成）、编号序列补齐、误剥表格行回接。

## 六、缓存与可复现性

API 结果按图像内容哈希落盘（`cache/`、上采样 tile 存 `cache_up/`），本地格级 OCR 存 `rec_cache.sqlite`，几何校准逐行 OCR 存 `cache_geo/`。作用：省额度、重跑只补失败项、让重放稳定。干净环境首次运行缓存为空，会真实调用 API 建立。

**错误响应不入缓存**——服务端偶发吐 HTML 错误页 / 错误信封 / 复读退化（同一行吐上百遍），一律判失败并重试，绝不写缓存污染后续重放。

FinixDoc-VL 是生成式模型，同一张图两次调用未必逐字相同。本工程把这种不确定性压到最小：几何决策（切哪、跳哪、行列骨架）全部由代码确定性给出，模型只负责识别内容，不合格的 tile 由本地确定性模型兜底。实测同一输入连续多次运行输出完全一致。
