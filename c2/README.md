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

**运行时长**：实测 24 核 / 32 并发 / 冷缓存，整批 100 张约 30 分钟（B 榜实测两个半边各约 10 分钟，本地小模型的残差重读与几何定级各加一点），远低于 3h 上限。API 限流时客户端退避重试；`API_TIMEOUT=40 / API_RETRIES=3` 按"快速放弃 + 靠缓存重跑补失败"调过——重跑时已成功的 tile 直接命中缓存，等于天然的末尾补扫。

## 三、合规声明

- **大模型**：全工程唯一的 VLM 接口是主办方提供的 FinixDoc-VL API，调用点集中在 `src/common/api_client.py`（`requests.post` 两处），无任何其它第三方模型 API、SDK 或网络地址。
- **Prompt**：该 API 的请求体只有 `userId / apiKey / fileName + 图片文件`，不接受文本 prompt 参数，因此本方案没有 prompt 配置文件——引导模型的工作全在图像侧（切在哪、放大多少、空白块跳过不送）。
- **本地小模型**：`PP-OCRv6_rec_small.onnx`，**5.27M 参数**（< 10M 上限），CPU + onnxruntime，用于 TABLE 残差格级重读与 LONG 几何标题定级。同引擎载入的 det/cls 模型调用时均置 `use_det=False / use_cls=False` 不参与推理，三者合计也仅 7.72M。模型随 `rapidocr` wheel 安装，运行期不联网。
- **无硬编码**：不含针对特定测试图片的固定输出、白名单或 uuid 分支；所有阈值都是几何/统计维度的通用判据，集中在 `common/config.py` 与 `table/tiles.py`。

## 四、目录结构

```
c2/
├── run.sh                    一键端到端脚本(唯一推荐入口)
├── init_env.sh               conda 环境初始化
├── requirements.txt          依赖清单
├── plan.md                   方案演进记录
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
│   │   ├── slicer_long.py      水平投影找行间空白带下刀,不劈断字行
│   │   ├── stitch_long.py      接缝拼接:跨条重复行去重、断表跨条重连
│   │   ├── heading_norm.py     标题层级校正:栈模型 + 编号序列 + 目录/封面题名
│   │   ├── geom_heading.py     几何标题定级:回原图量字号/淡横线做全局裁决
│   │   └── table_fix.py        长文内表格自愈:满宽横幅行 colspan 归一
│   ├── table/                TABLE(大表图)流水线
│   │   ├── run_table.py        入口 parse_table:crop → ocr → merge
│   │   ├── crop.py             Stage I 纯几何裁剪:剥页眉页脚水印、切并排子表
│   │   ├── geom.py             共享几何原语:投影分段、框线检测、并排缝
│   │   ├── slicer_table.py     在网格线处切分,空白 tile 预判跳过
│   │   ├── tiles.py            tile 公共层:尺寸/上采样策略、并发调用
│   │   ├── grid_ocr.py         Stage II 骨架 OCR:墨迹几何做尺子,零容差判定
│   │   ├── cell_ocr.py         残差修复:不一致 tile 回落 PP-OCRv6s 逐格重读
│   │   └── stitch_single.py    Stage III 2D 重组 → 完整 <table>
│   ├── metrics/              本地评测(复现官方三指标)
│   └── tools/                开发期审计/评测工具(见 tools/README.md)
└── out/                      运行产出 —— 未纳入版本管理
```

运行期产物（`cache/`、`cache_up/`、`cache_geo/`、`rec_cache.sqlite`、`out/`、`data/`）均已 gitignore，只存在于本地。

## 五、方法概要

两类图的痛点不同，走两条独立流水线。

### LONG（面条图，长宽比 > 30）

单图高达数万像素，整图送模型必然 OOM / 长上下文失效。

1. **切**——水平投影求每行墨量，在行间空白带下刀。切点落在字行之间，接缝处不出现半行残字，这是 ReadOrder 与 TextEdit 的主要失分源。
2. **读**——各条带并发送 API，按内容哈希缓存。
3. **拼**——接缝重叠区去重；长表格中部没有空白带、切点必然落在表内，上条收口成 `</table>`、下条重开 `<table>`，此处做跨条表格重连。
4. **正**——模型按条带识别，`#` 是局部判断，拼成整篇后同一编号序列常在条带边界被重置。`heading_norm` 用栈模型 + 编号相对递减 + 序列历史纠漂移，封面标题提升 L1、目录伪标题降级；`table_fix` 把表内满宽横幅行的 `colspan` 归一。
5. **定级**——`geom_heading` 是最后一步，也是唯一回原图做全局裁决的一步。层级本质是排版信息，前面几步都只能从文本推断；这里回原图逐行量标题**字号**与上下**淡横线**，重定 `#` 层级。必须放在拼接之后——整篇拼好了字号才有全局可比性。逐行识别用本地 PP-OCRv6s，按图缓存到 `cache_geo/`。

### TABLE（大表图，单页可达 2 亿像素）

密集网格表，模型读整表会跳行/复读——自回归解码靠自己的输出史追踪表内位置，重复内容摧毁这种追踪。

1. **Stage I 裁剪**（纯几何，不调模型）——剥离表外 furniture（判据：矮 + 窄 + 非最大连通块），再按横线断裂与列模式周期切分并排子表。竖线默认是列分隔而非分表信号。
2. **Stage II 骨架 OCR**——严格在网格线处切 tile，小字先上采样；墨量极低的空白 tile 直接跳过不送模型（送了就是幻觉）。核心是"一把尺、一个判定、一个修复"：墨迹几何给出期望行数与每行内容宽 → 零容差恒等判定（禁众数、禁 ±1 容差）→ 不合格才触发修复。
3. **残差修复**——仍不一致的 tile 回落本地 PP-OCRv6s 逐格重读。行列位置由几何骨架给定，模型只回答"这格是什么"，位置控制权收回代码，**计数类错误（爆行/漏行/跳读）结构性不可能发生**。单调重复区（`3000` × N）恰是 API 最易失稳、逐格识别最容易的地方。
4. **Stage III 2D 重组**——tile 结果按行列装配成完整 `<table>`，做列校准与 colspan 审计。

## 六、缓存与可复现性

API 结果按图像内容哈希落盘（`cache/`、上采样 tile 存 `cache_up/`），本地格级 OCR 存 `rec_cache.sqlite`，几何定级逐行 OCR 存 `cache_geo/`。作用：省额度、重跑只补失败项、让重放稳定。干净环境首次运行缓存为空，会真实调用 API 建立。

**错误响应不入缓存**——服务端偶发吐 HTML 错误页 / 错误信封 / 复读退化（同一行吐上百遍），一律判失败并重试，绝不写缓存污染后续重放。

FinixDoc-VL 是生成式模型，同一张图两次调用未必逐字相同。本工程把这种不确定性压到最小：几何决策（切哪、跳哪、行列骨架）全部由代码确定性给出，模型只负责识别内容，不合格的 tile 由本地确定性模型兜底。实测同一输入连续多次运行输出完全一致。
