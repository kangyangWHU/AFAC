"""新 parse:PaddleOCR-VL(vLLM) 逐页解析 PDF → 每页 markdown。
- 无文本层兜底(实测兜底=乱序拼接,更差),完全以 VLM 的版面+阅读顺序为准。
- 表格:规整表转 markdown;含合并单元格(colspan/rowspan)的保留 HTML(md 表达不了)。
- 后处理:剥非语义属性、去 VLM 噪声(补充plane emoji、相邻重复行)。

运行环境:conda env `paddleocr`(有 PaddleOCRVL 客户端),连 vLLM 服务(env `vllm-paddleocr`)。
"""
from __future__ import annotations
import os
import re
import tempfile
from html.parser import HTMLParser

import fitz  # PyMuPDF 仅用于渲染页面成图,不参与解析

DEFAULT_SERVER = "http://127.0.0.1:8118/v1"
DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.5"

_PIPE = None


def _pipe(server_url: str, model_name: str, max_concurrency: int = 16):
    global _PIPE
    if _PIPE is None:
        from paddleocr import PaddleOCRVL
        _PIPE = PaddleOCRVL(pipeline_version="v1.5", vl_rec_backend="vllm-server",
                            vl_rec_server_url=server_url, vl_rec_api_model_name=model_name,
                            vl_rec_max_concurrency=max_concurrency)
    return _PIPE


# ---------- 渲染 + 推理 ----------

def _render_png(doc, pno: int, dpi: int, out_dir: str) -> str:
    path = os.path.join(out_dir, f"p{pno}.png")
    doc[pno].get_pixmap(dpi=dpi).save(path)
    return path


def _predict_markdown(pipe, png_path: str, out_dir: str) -> str:
    """PaddleOCRVL 结果落地为 md 再读回(结果对象不稳定暴露字符串,落地最稳)。"""
    sub = os.path.join(out_dir, "md")
    os.makedirs(sub, exist_ok=True)
    for res in pipe.predict(png_path):
        res.save_to_markdown(save_path=sub)
    mds = [f for f in os.listdir(sub) if f.endswith(".md")]
    if not mds:
        return ""
    text = open(os.path.join(sub, mds[0]), encoding="utf-8").read()
    for f in mds:
        os.remove(os.path.join(sub, f))
    return text


# ---------- 后处理 ----------

_ATTR = re.compile(r"\s+(?:style|border|class|align|valign|width|height)=(\"[^\"]*\"|'[^']*')")
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF]")          # 补充平面 emoji(🔥等),BMP 符号(❖)保留
_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
# LaTeX 噪声:PaddleOCR-VL 把下划线/上标渲成 $\underline{..}$ / $^{1}$,清成纯文本
_LATEX_CMD = re.compile(r"\\(?:underline|textbf|textit|text|mathrm|mathbf|emph|boldsymbol)\s*\{([^{}]*)\}")
_LATEX_SUP = re.compile(r"[\^_]\{([^{}]*)\}")


def _strip_latex(s: str) -> str:
    for _ in range(3):                       # 迭代剥嵌套 \underline{\text{..}}
        ns = _LATEX_CMD.sub(r"\1", s)
        if ns == s:
            break
        s = ns
    s = _LATEX_SUP.sub(r"\1", s)
    return s.replace("$", "")


class _Table(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row = None
        self._cell = None
        self.merged = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            if a.get("colspan") or a.get("rowspan"):
                self.merged = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _table_to_md(html: str) -> str:
    """规整表 → markdown 表;含合并单元格 → 返回剥过属性的 HTML(保结构)。"""
    p = _Table()
    p.feed(html)
    rows = [r for r in p.rows if r]
    if not rows:
        return _ATTR.sub("", html)
    if p.merged:
        return _ATTR.sub("", html)                       # md 无法表达合并单元格
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    esc = lambda c: c.replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(esc(c) for c in rows[0]) + " |",
           "| " + " | ".join(["---"] * w) + " |"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def clean_page(md: str) -> str:
    md = _TABLE.sub(lambda m: _table_to_md(m.group(0)), md)   # 表格先转(转换内部已剥属性)
    md = _ATTR.sub("", md)                                    # 表外残留属性再剥
    md = _EMOJI.sub("", md)
    md = _strip_latex(md)                                     # 清 $\underline{}$ / $^{1}$ 等
    # 去相邻完全重复行(VLM 偶发重复)
    out, prev = [], None
    for ln in md.splitlines():
        s = ln.strip()
        if s and s == prev:
            continue
        out.append(ln)
        if s:
            prev = s
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ---------- 对外接口 ----------

def parse_pdf(pdf_path: str, dpi: int = 200, max_concurrency: int = 16,
              server_url: str = DEFAULT_SERVER, model_name: str = DEFAULT_MODEL) -> list[dict]:
    """返回 [{'page_no': 1, 'markdown': '...'}, ...]"""
    pipe = _pipe(server_url, model_name, max_concurrency)
    doc = fitz.open(pdf_path)
    pages = []
    with tempfile.TemporaryDirectory() as tmp:
        for pno in range(doc.page_count):
            png = _render_png(doc, pno, dpi, tmp)
            raw = _predict_markdown(pipe, png, tmp)
            os.remove(png)
            pages.append({"page_no": pno + 1, "markdown": clean_page(raw)})
    doc.close()
    return pages


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default=None, help="输出目录,每页一个 p{n}.md")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("-s", "--start", type=int, default=1)
    ap.add_argument("-e", "--end", type=int, default=None)
    args = ap.parse_args()
    pages = parse_pdf(args.pdf, dpi=args.dpi)
    if args.end:
        pages = [p for p in pages if args.start <= p["page_no"] <= args.end]
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for p in pages:
            open(os.path.join(args.out, f"p{p['page_no']}.md"), "w", encoding="utf-8").write(p["markdown"])
        print(f"wrote {len(pages)} pages -> {args.out}")
    else:
        for p in pages:
            print(f"\n===== p.{p['page_no']} =====\n{p['markdown']}")
