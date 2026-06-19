# -*- coding: utf-8 -*-
"""读取 out/teds_table_full.json → 画 TEDS 分布图 + 短板分析表。"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 尽量挂一个中文字体（缺失则退英文标签，不报错）
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False

from config import OUT_DIR

d = json.load(open(os.path.join(OUT_DIR, "teds_table_full.json"), encoding="utf-8"))
rows = d["rows"]
teds = np.array([r["teds"] for r in rows])
mean = teds.mean()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# 左：直方图
ax[0].hist(teds, bins=20, range=(0, 1), color="#4C72B0", edgecolor="white")
ax[0].axvline(mean, color="crimson", ls="--", lw=2, label=f"mean={mean:.4f}")
ax[0].axvline(np.median(teds), color="green", ls=":", lw=2, label=f"median={np.median(teds):.4f}")
ax[0].set_xlabel("TEDS"); ax[0].set_ylabel("张数")
ax[0].set_title(f"TABLE 全量 {len(rows)} 张 TEDS 分布")
ax[0].legend()

# 右：排序后的累积曲线（看尾部）
xs = np.sort(teds)
ax[1].plot(range(1, len(xs) + 1), xs, marker=".", color="#55A868")
ax[1].axhline(mean, color="crimson", ls="--", lw=1)
for thr in (0.5, 0.7, 0.9):
    n = int((teds < thr).sum())
    ax[1].axhline(thr, color="gray", ls=":", lw=0.7)
    ax[1].text(2, thr + 0.005, f"<{thr}: {n}张", fontsize=9, color="gray")
ax[1].set_xlabel("排名（升序）"); ax[1].set_ylabel("TEDS")
ax[1].set_title("TEDS 排序曲线（看长尾短板）")

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "teds_dist.png")
plt.savefig(out_png, dpi=120)
print(f"已保存 {out_png}")

# 分桶统计
buckets = [(0, .3), (.3, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)]
print("\n=== 分桶 ===")
for lo, hi in buckets:
    n = int(((teds >= lo) & (teds < hi)).sum())
    print(f"  [{lo:.2f},{hi:.2f}): {n:>2} 张  {'█'*n}")

print(f"\n=== 最差 15 张 ===")
hdr = f"{'uuid':10} {'TEDS':>6} {'txt':>5} {'grid':>5} {'up':>4} {'tile':>7} {'skel':>9} {'lenR':>5} {'gt(tab/tr/td)':>14}"
print(hdr)
for r in rows[:15]:
    print(f"{r['uuid'][:8]:10} {r['teds']:6.3f} {r['text_score']:5.1f} "
          f"{str(r['grid_ok']):>5} {r['upsample']:>4} {r['tile_grid']:>7} "
          f"{r['skeleton']:>9} {r['len_ratio']:>5} "
          f"{str(r['gt_tables'])+'/'+str(r['gt_tr'])+'/'+str(r['gt_td']):>14}")
