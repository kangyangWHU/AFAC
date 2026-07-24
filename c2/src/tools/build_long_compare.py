# -*- coding: utf-8 -*-
"""逐文档对照页：左原图 / 右文本，比例同步滚动。

两种取文本的方式:

  # 榜单提交(无 GT)——右栏只有预测
  python src/tools/build_long_compare.py \
      --csv out/submission_long_B_scorefix_v4.csv \
      --images "data/AFACB榜评测数据集/finix_huge_long_rest_B/images" \
      --out out/long_table_audit.html

  # 训练集(有 GT)——右栏可在 GT / 官方 pred 间切换
  python src/tools/build_long_compare.py \
      --md-dir "data/AFAC 训练数据集_extracted/finixdocbench_huge_long_100/mds" \
      --images "data/AFAC 训练数据集_extracted/finixdocbench_huge_long_100/images" \
      --out out/long_train_gt.html

--csv 模式按 file_name ∩ images/ 取交集，故直接喂完整 submission 也只出 LONG 那一半。
--md-dir 模式把 xxx.md 收成 GT、xxx_pred.md 收成 PRED，同名图片配对。
图片不内嵌(长图动辄数 MB)，用相对路径引用，页面须留在 out/ 下就地打开。
"""
import argparse
import csv
import sys
import json
import os
from pathlib import Path
from urllib.parse import quote

csv.field_size_limit(sys.maxsize)      # 表格单元格可达数十万字符,默认 128K 会炸

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>LONG B 对照</title>
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
header button:hover { background: #414852; }
header label { display: flex; align-items: center; gap: 5px; user-select: none; }
.meta { color: #8b929c; margin-left: auto; }
main { flex: 1; display: grid; grid-template-columns: 1fr 6px 1fr; min-height: 0; }
.pane { overflow: auto; min-height: 0; }
#left { background: #101215; text-align: center; }
#left img { display: block; margin: 0 auto; }
#gutter { background: #3a3e44; cursor: col-resize; }
#right { background: #fbfbf9; color: #16181c; padding: 26px 34px 60vh; }
#right.raw { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
             white-space: pre-wrap; font-size: 13px; line-height: 1.55; }
#right h1, #right h2, #right h3, #right h4, #right h5, #right h6 {
  margin: 1.5em 0 .5em; line-height: 1.35; scroll-margin-top: 10px; }
#right h1 { font-size: 1.5em; } #right h2 { font-size: 1.3em; }
#right h3 { font-size: 1.15em; } #right h4 { font-size: 1.03em; }
#right h5, #right h6 { font-size: .97em; color: #444; }
#right p { margin: .55em 0; }
#right table { border-collapse: collapse; margin: 1em 0; font-size: .92em; }
#right td, #right th { border: 1px solid #999; padding: 3px 7px; }
.lv { display: inline-block; min-width: 2.1em; margin-right: .55em; padding: 0 4px;
      border-radius: 3px; font: 600 11px/1.7 ui-monospace, monospace;
      vertical-align: middle; color: #fff; background: #8b929c; }
.lv1 { background: #c0392b; } .lv2 { background: #d68910; } .lv3 { background: #1e8449; }
.lv4 { background: #2471a3; } .lv5 { background: #6c3483; } .lv6 { background: #566573; }
body.nolv .lv { display: none; }
mark { background: #ffe066; }
</style>
<header>
  <button id="prev">◀</button>
  <select id="pick"></select>
  <button id="next">▶</button>
  <select id="src" hidden></select>
  <label><input type="checkbox" id="sync" checked> 同步滚动</label>
  <label><input type="checkbox" id="lv" checked> 标题层级</label>
  <label><input type="checkbox" id="raw"> 原文</label>
  <label>缩放 <input type="range" id="zoom" min="20" max="200" value="100" style="width:110px"></label>
  <span class="meta" id="meta"></span>
</header>
<main>
  <div class="pane" id="left"><img id="img" alt=""></div>
  <div id="gutter"></div>
  <div class="pane" id="right"></div>
</main>
<script>
const DOCS = __DOCS__;
const SRCS = __SRCS__;
const $ = s => document.querySelector(s);
const left = $('#left'), right = $('#right'), img = $('#img');
let idx = 0, cur = SRCS[0];

// --- 极简 markdown 渲染：标题 / HTML 表格原样 / 其余成段 -----------------
function render(md) {
  const out = [];
  const lines = md.split('\\n');
  let para = [], tbl = null;
  const flush = () => {
    if (para.length) { out.push('<p>' + esc(para.join('\\n')) + '</p>'); para = []; }
  };
  for (const line of lines) {
    if (tbl !== null) {
      tbl.push(line);
      if (/<\\/table>/i.test(line)) { out.push(tbl.join('\\n')); tbl = null; }
      continue;
    }
    if (/^\\s*<table\\b/i.test(line)) {
      flush();
      tbl = [line];
      if (/<\\/table>/i.test(line)) { out.push(tbl.join('\\n')); tbl = null; }
      continue;
    }
    const h = line.match(/^(#{1,6})\\s+(.*)$/);
    if (h) {
      flush();
      const n = h[1].length;
      out.push(`<h${n}><span class="lv lv${n}">H${n}</span>${esc(h[2])}</h${n}>`);
      continue;
    }
    if (!line.trim()) { flush(); continue; }
    para.push(line);
  }
  flush();
  if (tbl) out.push(tbl.join('\\n'));   // 未闭合的表格也别吞掉
  return out.join('\\n');
}
function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function show(i, keepScroll) {
  idx = (i + DOCS.length) % DOCS.length;
  const d = DOCS[idx];
  const md = d.srcs[cur] || '';
  const keep = keepScroll ? [left.scrollTop, frac(right)] : null;
  $('#pick').value = String(idx);
  img.src = d.img;
  right.classList.toggle('raw', $('#raw').checked);
  right.innerHTML = $('#raw').checked ? esc(md) : render(md);
  const heads = (md.match(/^#{1,6} /gm) || []).length;
  const tables = (md.match(/<table\\b/gi) || []).length;
  const other = SRCS.filter(k => k !== cur)
    .map(k => ` · vs ${k} ${sign((d.srcs[k] || '').length - md.length)} 字`).join('');
  $('#meta').textContent = `${idx + 1}/${DOCS.length} · ${cur} · `
    + `${md.length.toLocaleString()} 字 · ${heads} 标题 · ${tables} 表${other}`;
  // 切换来源时保持视线不动，切换文档时归零
  if (keep) { left.scrollTop = keep[0];
              right.scrollTop = keep[1] * (right.scrollHeight - right.clientHeight); }
  else { left.scrollTop = right.scrollTop = 0; }
  location.hash = d.name;
}
const sign = n => (n > 0 ? '+' : '') + n.toLocaleString();

// --- 比例同步滚动 ------------------------------------------------------
let lock = false;
const frac = el => {
  const room = el.scrollHeight - el.clientHeight;
  return room > 0 ? el.scrollTop / room : 0;
};
const link = (src, dst) => src.addEventListener('scroll', () => {
  if (!$('#sync').checked || lock) return;
  lock = true;
  dst.scrollTop = frac(src) * (dst.scrollHeight - dst.clientHeight);
  requestAnimationFrame(() => { lock = false; });
});
link(left, right); link(right, left);

// --- 控件 --------------------------------------------------------------
$('#pick').innerHTML = DOCS.map((d, i) => `<option value="${i}">${i + 1}. ${d.name}</option>`).join('');
$('#pick').onchange = e => show(+e.target.value);
$('#prev').onclick = () => show(idx - 1);
$('#next').onclick = () => show(idx + 1);
$('#raw').onchange = () => show(idx, true);
if (SRCS.length > 1) {
  const sel = $('#src');
  sel.hidden = false;
  sel.innerHTML = SRCS.map(k => `<option>${k}</option>`).join('');
  sel.onchange = e => { cur = e.target.value; show(idx, true); };
}
$('#lv').onchange = e => document.body.classList.toggle('nolv', !e.target.checked);
$('#zoom').oninput = e => { img.style.width = e.target.value + '%'; };
img.style.width = '100%';
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') show(idx - 1);
  if (e.key === 'ArrowRight') show(idx + 1);
  if (e.key === 's' && SRCS.length > 1) {      // 原地 A/B 闪切，视线不动
    cur = SRCS[(SRCS.indexOf(cur) + 1) % SRCS.length];
    $('#src').value = cur;
    show(idx, true);
  }
});

// 拖动分栏
$('#gutter').addEventListener('mousedown', e => {
  e.preventDefault();
  const move = ev => {
    const p = Math.min(85, Math.max(15, ev.clientX / window.innerWidth * 100));
    document.querySelector('main').style.gridTemplateColumns = `${p}% 6px 1fr`;
  };
  const up = () => { removeEventListener('mousemove', move); removeEventListener('mouseup', up); };
  addEventListener('mousemove', move); addEventListener('mouseup', up);
});

const start = DOCS.findIndex(d => d.name === decodeURIComponent(location.hash.slice(1)));
show(start >= 0 ? start : 0);
</script>
"""


IMG_EXT = (".jpg", ".jpeg", ".png")


def _from_csv(csv_path, names):
    """提交 CSV → {图名: {"PRED": md}}。ground_truth 列在提交格式里装的是预测。"""
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        return {r["file_name"]: {"PRED": r["ground_truth"]}
                for r in csv.DictReader(f) if r["file_name"] in names}


def _from_md_dir(md_dir, names):
    """训练集 mds/ → {图名: {"GT": md, "PRED": md}}。xxx.md 是 GT、xxx_pred.md 是官方预测。"""
    stem_to_img = {Path(n).stem: n for n in names}
    out = {}
    for p in sorted(Path(md_dir).iterdir()):
        if p.suffix != ".md":
            continue
        stem, key = (p.stem[:-5], "PRED") if p.stem.endswith("_pred") else (p.stem, "GT")
        img = stem_to_img.get(stem)
        if img:
            out.setdefault(img, {})[key] = p.read_text(encoding="utf-8")
    return out


def build(csv_path, md_dir, image_dir, out_path):
    image_dir = Path(image_dir).resolve()
    out_path = Path(out_path).resolve()
    names = {p.name for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT}

    by_img = _from_md_dir(md_dir, names) if md_dir else _from_csv(csv_path, names)

    docs = [{"name": n,
             "img": quote(os.path.relpath(image_dir / n, out_path.parent)
                          .replace(os.sep, "/")),
             "srcs": by_img[n]}
            for n in sorted(by_img)]
    # GT 在前:打开就是基准，按 s 闪切到 PRED
    srcs = [k for k in ("GT", "PRED") if any(k in d["srcs"] for d in docs)]

    missing = sorted(names - set(by_img))
    if missing:
        print(f"[warn] {len(missing)} 张图没有配到文本: {missing[:3]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        PAGE.replace("__DOCS__", json.dumps(docs, ensure_ascii=False))
            .replace("__SRCS__", json.dumps(srcs)),
        encoding="utf-8")
    print(f"Wrote {out_path} ({len(docs)} docs, {'/'.join(srcs)}, "
          f"{out_path.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="提交 CSV(file_name, ground_truth)")
    g.add_argument("--md-dir", help="训练集 mds/ 目录(GT + _pred)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.csv, a.md_dir, a.images, a.out)


if __name__ == "__main__":
    main()
