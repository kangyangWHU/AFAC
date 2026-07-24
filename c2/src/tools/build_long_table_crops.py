# -*- coding: utf-8 -*-
"""把 LONG 文档里的表格单独裁出来对照：左=原图裁片，右=重构的 <table>。

长文里的表格散在几万 px 的面条图中间,逐张翻很费劲,而表格分权重高。
这里用 slicer_long.table_bands 的框线检测定位表格区,裁成独立图片,
与预测里的 <table> 按文档序并排。

**数量不符是主要审计信号**:框线检测到 N 个表格带、预测里只有 M 个 <table>,
N≠M 就是漏读/合并/过切。注意两边都不是绝对真值 ——
无框表检测不到(带数偏少),跨页续表会被切成多段(带数偏多),所以不符只是"要看一眼",
不是"一定错"。

用法(在 src 目录)：
  python -m tools.build_long_table_crops --csv ../out/_long_B_v8_raw.csv \
      --images "../data/AFACB榜评测数据集/finix_huge_long_rest_B/images" \
      --out ../out/long_table_crops.html
"""
import os
import re
import csv
import sys
import json
import shutil
import argparse
from pathlib import Path
from multiprocessing import Pool
from urllib.parse import quote

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from common.preprocess import prep
from long.slicer_long import table_bands, slice_long
from common.api_client import _cache_path, _img_bytes

csv.field_size_limit(sys.maxsize)

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.I | re.S)
MARGIN = 120          # 裁片上下留白,能看到表格前后一两行文字(判断表头有没有掉出去)
MAX_W = 760           # 裁片宽度上限,原图 ~1500 → 半宽,够看清且文件小


def _crop_one(job):
    """裁出每个表格带,并取回**该带所在条带**的重构内容。

    右栏只放 <table> 的话,"图上有表、我们没读成表"这种最该看的情形反而什么都看不到。
    改成给出该区域对应条带的完整重构 —— 漏读的表会以正文形式露出来。
    """
    name, img_path, crop_dir = job
    im = prep(Image.open(img_path))
    w, h = im.size
    strips, cuts = slice_long(im, target_h=5000)
    outs = []
    for s in strips:
        p = _cache_path(_img_bytes(s)[0])
        outs.append(open(p, encoding="utf-8").read() if os.path.exists(p) else "")

    res = []
    for k, (y0, y1) in enumerate(table_bands(im)):
        a, b = max(0, y0 - MARGIN), min(h, y1 + MARGIN)
        c = im.crop((0, a, w, b))
        if w > MAX_W:
            c = c.resize((MAX_W, max(1, int(c.size[1] * MAX_W / w))), Image.LANCZOS)
        fn = f"{Path(name).stem}_{k}.jpg"
        c.convert("RGB").save(os.path.join(crop_dir, fn), quality=80)
        # 与该带在 y 上有交叠的条带 = 这段图的重构来源
        idx = [i for i in range(len(strips)) if cuts[i] < y1 and cuts[i + 1] > y0]
        res.append({"file": fn, "y0": y0, "y1": y1, "h": y1 - y0, "strips": idx,
                    "recon": _focus("\n\n".join(outs[i] for i in idx))})
    return name, res


CONTEXT = 4               # 表格前后各留几行:够判断表头/表注有没有掉到表外


def _focus(md):
    """把整条带的重构收敛到表格附近。

    直接摊开整条带会把表埋进几千字释义正文里(条带高 5000px,内容远多于表)。
    这里只留 <table> 及其前后各 CONTEXT 行;整段一张表都没有时才全给 ——
    那正是"图上是表、我们读成了正文"的情形,需要看正文才判断得了。
    """
    lines = [l for l in (md or "").split("\n")]
    spans = []                                   # 表格占用的行区间
    start = None
    for i, l in enumerate(lines):
        if start is None and re.search(r"<table\b", l, re.I):
            start = i
        if start is not None and re.search(r"</table>", l, re.I):
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(lines) - 1))
    if not spans:
        return md

    keep = set()
    for a, b in spans:
        keep.update(range(max(0, a - CONTEXT), min(len(lines), b + CONTEXT + 1)))
    out, prev = [], None
    for i in sorted(keep):
        if prev is not None and i > prev + 1:
            out.append("⋯⋯")                     # 标出被略去的段落
        out.append(lines[i])
        prev = i
    return "\n".join(out)


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>LONG 内嵌表格对照</title>
<style>
* { box-sizing: border-box; }
body { font: 14px/1.6 system-ui, "Noto Sans CJK SC", sans-serif; margin: 0;
       background: #1b1d20; color: #e8e8e8; }
header { position: sticky; top: 0; z-index: 5; padding: 8px 14px; background: #26292d;
         border-bottom: 1px solid #3a3e44; display: flex; gap: 14px; align-items: center; }
header select { background: #14161a; color: #e8e8e8; border: 1px solid #464b52;
                padding: 4px 6px; font: inherit; max-width: 50ch; }
.doc { border-bottom: 3px solid #3a3e44; padding: 10px 14px 22px; }
.doc h2 { font-size: 15px; margin: 6px 0 10px; font-weight: 600; }
.bad { color: #ff8a80; }
.ok { color: #8bc34a; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start;
        margin-bottom: 18px; }
.pane { background: #101215; border: 1px solid #3a3e44; max-height: 78vh; overflow: auto; }
.pane img { display: block; width: 100%; }
.tbl { background: #fbfbf9; color: #16181c; padding: 12px; }
.tbl table { border-collapse: collapse; font-size: 12px;
              width: 100%; table-layout: fixed; }   /* 各表列宽一致,便于跨片比对 */
.tbl td, .tbl th { border: 1px solid #999; padding: 2px 5px;
                   word-break: break-word; vertical-align: top; }
.tbl table + table { margin-top: 6px; border-top: 3px solid #c0392b; }  /* 表与表的分界 */
.tbl .txt { color: #444; font-size: 12px; margin: 2px 0; }
.tbl b { display: block; margin: 8px 0 3px; }
.cap { font-size: 12px; color: #8b929c; padding: 3px 6px; background: #26292d; }
.none { padding: 14px; color: #8b929c; }
</style>
<header>
  <select id="pick"></select>
  <label><input type="checkbox" id="onlybad"> 只看数量不符</label>
  <span id="meta"></span>
</header>
<div id="body"></div>
<script>
const DOCS = __DOCS__;
const $ = s => document.querySelector(s);
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function renderRecon(md){
  if(!md) return '<div class="none">— 该条带无缓存 —</div>';
  const out=[]; let tbl=null;
  for(const line of md.split('\n')){
    if(tbl!==null){ tbl.push(line); if(/<\/table>/i.test(line)){out.push(tbl.join('\n'));tbl=null;} continue; }
    if(/^\s*<table\b/i.test(line)){ tbl=[line]; if(/<\/table>/i.test(line)){out.push(tbl.join('\n'));tbl=null;} continue; }
    const h=line.match(/^(#{1,6})\s+(.*)$/);
    if(h){ out.push(`<b>${'#'.repeat(h[1].length)} ${esc(h[2])}</b>`); continue; }
    if(line.trim()) out.push('<div class="txt">'+esc(line)+'</div>');
  }
  if(tbl) out.push(tbl.join('\n'));
  return out.join('\n');
}
function render(){
  const only = $('#onlybad').checked;
  const list = only ? DOCS.filter(d => d.crops.length !== d.tables.length) : DOCS;
  $('#body').innerHTML = list.map((d,i)=>{
    const bad = d.crops.length !== d.tables.length;
    const n = Math.max(d.crops.length, d.tables.length);
    let rows = '';
    for (let k=0;k<n;k++){
      const c = d.crops[k], t = d.tables[k];
      const right = c ? renderRecon(c.recon)
                      : (t ? `<div class="cap">预测第 ${k+1} 张表(图上无对应带)</div>`+t
                           : '<div class="none">—</div>');
      rows += `<div class="pair">
        <div class="pane">${c ? `<div class="cap">裁片 ${k+1}/${d.crops.length} · y ${c.y0}–${c.y1} (${c.h}px) · 条带 ${c.strips.join(',')}</div><img loading="lazy" src="${d.dir}/${c.file}">` : '<div class="none">— 图上没有检出对应表格带 —</div>'}</div>
        <div class="pane tbl">${right}</div>
      </div>`;
    }
    return `<div class="doc" id="d${i}"><h2>${esc(d.name)}
      <span class="${bad?'bad':'ok'}">带 ${d.crops.length} / 预测表 ${d.tables.length}</span></h2>${rows}</div>`;
  }).join('');
  $('#pick').innerHTML = list.map((d,i)=>`<option value="d${i}">${d.crops.length===d.tables.length?'　':'⚠ '}${d.name}</option>`).join('');
  $('#meta').textContent = `${list.length} 份 · 数量不符 ${DOCS.filter(d=>d.crops.length!==d.tables.length).length}`;
}
$('#pick').onchange = e => document.getElementById(e.target.value).scrollIntoView();
$('#onlybad').onchange = render;
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    out_path = Path(a.out).resolve()
    crop_dir = out_path.parent / (out_path.stem + "_img")
    if crop_dir.exists():
        shutil.rmtree(crop_dir)                 # 重跑时清干净,免得留下上一版的裁片
    crop_dir.mkdir(parents=True)

    img_dir = Path(a.images).resolve()
    with open(a.csv, encoding="utf-8", newline="") as f:
        preds = {r["file_name"]: r["ground_truth"] for r in csv.DictReader(f)}
    names = sorted(n for n in os.listdir(img_dir) if n in preds)

    jobs = [(n, str(img_dir / n), str(crop_dir)) for n in names]
    with Pool(a.workers) as pool:
        crops = dict(pool.map(_crop_one, jobs))

    docs = []
    for n in names:
        c, t = crops[n], _TABLE_RE.findall(preds[n])
        if c or t:
            docs.append({"name": n, "dir": quote(crop_dir.name),
                         "crops": c, "tables": t})

    out_path.write_text(PAGE.replace("__DOCS__", json.dumps(docs, ensure_ascii=False)),
                        encoding="utf-8")
    bad = sum(1 for d in docs if len(d["crops"]) != len(d["tables"]))
    print(f"{len(docs)} 份含表 · 裁片 {sum(len(d['crops']) for d in docs)} 张 · "
          f"预测表 {sum(len(d['tables']) for d in docs)} 张 · 数量不符 {bad} 份")
    print(f"→ {out_path}  (裁片在 {crop_dir}/)")


if __name__ == "__main__":
    main()
