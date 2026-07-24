# -*- coding: utf-8 -*-
"""标题改动审阅页:左=原图,右=渲染文档,每处改动挂徽章。

对比 旧预测(--old) 与 新预测(--new),只看**标题标记**的变化:
  旧是标题、新变正文  → 红色 DELETE 徽章(几何删除/伪标题降正文)
  旧是正文、新变标题  → 绿色 ADD Hx 徽章(编号序列推断补回)
  两边都是标题但层级变 → 黄色 H1→H2 徽章
正文内容本身不比对(两版只有 # 标记与少量层级差)。

对齐方式:把两版都剥掉 # 后做 difflib 行对齐 —— 文本相同、只有标记不同,
对齐是平凡的;同文重复行(3.2 如实告知义务 ×3)也按位置一一对应,不会串。

用法(在 src 目录):
  python -m tools.build_long_diff --old ../out/_long_B_v8_raw.csv \
      --new ../out/_long_B_v9_raw.csv \
      --images "../data/AFACB榜评测数据集/finix_huge_long_rest_B/images" \
      --out ../out/long_heading_diff.html
"""
import os
import re
import csv
import sys
import json
import difflib
import argparse
from pathlib import Path
from urllib.parse import quote

csv.field_size_limit(sys.maxsize)
_H = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def _parse(md):
    """→ [(level, text)] per non-empty line; level 0 = body."""
    out = []
    for ln in (md or "").splitlines():
        if not ln.strip():
            continue
        m = _H.match(ln)
        if m and m.group(2):
            out.append((len(m.group(1)), m.group(2)))
        else:
            out.append((0, ln.strip()))
    return out


def diff_marks(old_md, new_md):
    """Render lines = NEW doc; each gets a mark: ''|'add'|'del'|'chg'."""
    O, N = _parse(old_md), _parse(new_md)
    sm = difflib.SequenceMatcher(a=[t for _, t in O], b=[t for _, t in N],
                                 autojunk=False)
    marks = [""] * len(N)
    info = [""] * len(N)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            lo, ln_ = O[i + k][0], N[j + k][0]
            if lo == ln_:
                continue
            if lo > 0 and ln_ == 0:
                marks[j + k] = "del"; info[j + k] = f"was H{lo}"
            elif lo == 0 and ln_ > 0:
                marks[j + k] = "add"; info[j + k] = f"ADD H{ln_}"
            else:
                marks[j + k] = "chg"; info[j + k] = f"H{lo}→H{ln_}"
    return N, marks, info


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>标题改动审阅</title>
<style>
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { font: 14px/1.6 system-ui, "Noto Sans CJK SC", sans-serif; display: flex;
       flex-direction: column; background: #1b1d20; color: #e8e8e8; }
header { flex: none; display: flex; align-items: center; gap: 14px; padding: 8px 14px;
         background: #26292d; border-bottom: 1px solid #3a3e44; flex-wrap: wrap; }
header select { background: #14161a; color: #e8e8e8; border: 1px solid #464b52;
                padding: 4px 6px; font: inherit; max-width: 46ch; }
header button { background: #343941; color: #e8e8e8; border: 1px solid #4c525a;
                padding: 4px 11px; font: inherit; cursor: pointer; }
.meta { color: #8b929c; margin-left: auto; }
main { flex: 1; display: grid; grid-template-columns: 1fr 6px 1fr; min-height: 0; }
.pane { overflow: auto; min-height: 0; }
#left { background: #101215; text-align: center; }
#left img { display: block; margin: 0 auto; width: 100%; }
#gutter { background: #3a3e44; cursor: col-resize; }
#right { background: #fbfbf9; color: #16181c; padding: 24px 32px 60vh; }
#right h1,#right h2,#right h3,#right h4,#right h5,#right h6 {
  margin: 1.3em 0 .4em; line-height: 1.35; }
#right h1{font-size:1.45em}#right h2{font-size:1.25em}#right h3{font-size:1.12em}
#right h4{font-size:1.02em}#right h5,#right h6{font-size:.97em;color:#444}
#right p { margin: .5em 0; }
#right table { border-collapse: collapse; margin: .8em 0; font-size: .9em;
               width: 100%; table-layout: fixed; }
#right td, #right th { border: 1px solid #999; padding: 2px 6px; word-break: break-word; }
.lv { display:inline-block; min-width:2em; margin-right:.5em; padding:0 4px;
      border-radius:3px; font:600 11px/1.7 ui-monospace,monospace; color:#fff; background:#8b929c; }
.lv1{background:#c0392b}.lv2{background:#d68910}.lv3{background:#1e8449}
.lv4{background:#2471a3}.lv5{background:#6c3483}.lv6{background:#566573}
.badge { display:inline-block; margin-right:.5em; padding:0 6px; border-radius:3px;
         font:700 11px/1.8 ui-monospace,monospace; }
.b-del { background:#c62828; color:#fff; }
.b-add { background:#2e7d32; color:#fff; }
.b-chg { background:#f9a825; color:#000; }
.deleted-line { background:#fdecea; padding:2px 6px; border-left:3px solid #c62828; }
.added-head   { background:#e8f5e9; border-left:3px solid #2e7d32; padding-left:6px; }
.chg-head     { background:#fff8e1; border-left:3px solid #f9a825; padding-left:6px; }
</style>
<header>
  <button id="prev">◀</button>
  <select id="pick"></select>
  <button id="next">▶</button>
  <label><input type="checkbox" id="sync" checked> 同步滚动</label>
  <button id="jump">跳到下一处改动</button>
  <span class="meta" id="meta"></span>
</header>
<main>
  <div class="pane" id="left"><img id="img" alt=""></div>
  <div id="gutter"></div>
  <div class="pane" id="right"></div>
</main>
<script>
const DOCS = __DOCS__;
const $ = s => document.querySelector(s);
const left = $('#left'), right = $('#right'), img = $('#img');
let idx = 0;
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function render(d){
  const out=[]; let tbl=null;
  d.lines.forEach(([lv,txt],i)=>{
    const mark=d.marks[i], info=d.info[i];
    if(tbl!==null){ tbl.push(txt); if(/<\/table>/i.test(txt)){out.push(tbl.join('\n'));tbl=null;} return; }
    if(/^\s*<table\b/i.test(txt)){ tbl=[txt]; if(/<\/table>/i.test(txt)){out.push(txt);tbl=null;} return; }
    if(mark==='del'){
      out.push(`<p class="deleted-line"><span class="badge b-del">DELETE</span>`+
               `<span style="color:#8b929c;font-size:11px">${info}</span> ${esc(txt)}</p>`); return;
    }
    if(lv>0){
      const badge = mark==='add' ? `<span class="badge b-add">${info}</span>`
                  : mark==='chg' ? `<span class="badge b-chg">${info}</span>` : '';
      const cls = mark==='add' ? 'added-head' : mark==='chg' ? 'chg-head' : '';
      out.push(`<h${lv} class="${cls}">${badge}<span class="lv lv${lv}">H${lv}</span>${esc(txt)}</h${lv}>`);
      return;
    }
    out.push('<p>'+esc(txt)+'</p>');
  });
  if(tbl) out.push(tbl.join('\n'));
  return out.join('\n');
}
function show(i){
  idx=(i+DOCS.length)%DOCS.length;
  const d=DOCS[idx];
  $('#pick').value=String(idx);
  img.src=d.img;
  right.innerHTML=render(d);
  const na=d.marks.filter(m=>m==='add').length, nd=d.marks.filter(m=>m==='del').length,
        nc=d.marks.filter(m=>m==='chg').length;
  $('#meta').textContent=`${idx+1}/${DOCS.length} · ADD ${na} · DELETE ${nd} · 层级变 ${nc}`;
  left.scrollTop=right.scrollTop=0;
}
let lock=false;
const frac=el=>{const r=el.scrollHeight-el.clientHeight;return r>0?el.scrollTop/r:0;};
const link=(s,d)=>s.addEventListener('scroll',()=>{ if(!$('#sync').checked||lock)return;
  lock=true; d.scrollTop=frac(s)*(d.scrollHeight-d.clientHeight);
  requestAnimationFrame(()=>{lock=false;}); });
link(left,right); link(right,left);
$('#pick').innerHTML=DOCS.map((d,i)=>{
  const n=d.marks.filter(m=>m).length;
  return `<option value="${i}">${n?`[${n}] `:'　  '}${d.name}</option>`;}).join('');
$('#pick').onchange=e=>show(+e.target.value);
$('#prev').onclick=()=>show(idx-1);
$('#next').onclick=()=>show(idx+1);
$('#jump').onclick=()=>{
  const els=[...right.querySelectorAll('.deleted-line,.added-head,.chg-head')];
  const y=right.scrollTop+5;
  const nxt=els.find(e=>e.offsetTop>y+10);
  if(nxt) right.scrollTop=nxt.offsetTop-60;
};
$('#gutter').addEventListener('mousedown',e=>{e.preventDefault();
  const mv=ev=>{const p=Math.min(85,Math.max(15,ev.clientX/innerWidth*100));
    document.querySelector('main').style.gridTemplateColumns=`${p}% 6px 1fr`;};
  const up=()=>{removeEventListener('mousemove',mv);removeEventListener('mouseup',up);};
  addEventListener('mousemove',mv);addEventListener('mouseup',up);});
show(0);
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def read(p):
        with open(p, encoding="utf-8", newline="") as f:
            return {r["file_name"]: r["ground_truth"] for r in csv.DictReader(f)}
    old, new = read(a.old), read(a.new)
    out_path = Path(a.out).resolve()
    img_dir = Path(a.images).resolve()

    docs = []
    for n in sorted(new):
        if n not in old:
            continue
        lines, marks, info = diff_marks(old[n], new[n])
        rel = os.path.relpath(img_dir / n, out_path.parent)
        docs.append({"name": n, "img": quote(rel.replace(os.sep, "/")),
                     "lines": lines, "marks": marks, "info": info})
    docs.sort(key=lambda d: d["name"])           # 固定顺序(按文件名),每次渲染不变

    out_path.write_text(PAGE.replace("__DOCS__", json.dumps(docs, ensure_ascii=False)),
                        encoding="utf-8")
    na = sum(m == "add" for d in docs for m in d["marks"])
    nd = sum(m == "del" for d in docs for m in d["marks"])
    nc = sum(m == "chg" for d in docs for m in d["marks"])
    print(f"{len(docs)} docs · ADD {na} · DELETE {nd} · level-change {nc} → {out_path}")


if __name__ == "__main__":
    main()
