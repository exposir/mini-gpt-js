// probe-song.js —— 摸清宋诗的卷范围与实际产出率，再决定要不要全抓
const MIRRORS = [
  (f) => `https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/%E5%85%A8%E5%94%90%E8%AF%97/${f}`,
  (f) => `https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/${f}`,
];
async function head(file) {
  for (const mk of MIRRORS) {
    try {
      const res = await fetch(mk(file), { method: "HEAD", signal: AbortSignal.timeout(20000) });
      if (res.ok) return true;
    } catch { /* 换镜像 */ }
  }
  return false;
}
async function getJson(file) {
  for (const mk of MIRRORS) {
    try {
      const res = await fetch(mk(file), { signal: AbortSignal.timeout(30000) });
      if (res.ok) return await res.json();
    } catch { /* 换镜像 */ }
  }
  return null;
}

// 1) 二分找上界：宋诗卷号步长 1000
console.log("探测宋诗卷范围...");
let hi = 1000;
while (await head(`poet.song.${hi}.json`)) { hi *= 2; if (hi > 512000) break; }
let lo = hi / 2;
while (lo + 1000 < hi) {
  const mid = Math.floor((lo + hi) / 2 / 1000) * 1000;
  if (await head(`poet.song.${mid}.json`)) lo = mid; else hi = mid;
}
console.log(`宋诗最大卷号 ≈ ${lo}  →  约 ${lo / 1000 + 1} 卷`);

// 2) 抽 3 卷看产出率（四体裁筛选后能留下多少）
const SEG5 = /^[\u4e00-\u9fff]{5}，[\u4e00-\u9fff]{5}。$/;
const SEG7 = /^[\u4e00-\u9fff]{7}，[\u4e00-\u9fff]{7}。$/;
const FORMS = [
  { name: "五绝", segs: 2, re: SEG5 }, { name: "七绝", segs: 2, re: SEG7 },
  { name: "五律", segs: 4, re: SEG5 }, { name: "七律", segs: 4, re: SEG7 },
];
let raw = 0, kept = 0;
const stats = { 五绝: 0, 七绝: 0, 五律: 0, 七律: 0 };
for (const v of [0, Math.floor(lo / 2 / 1000) * 1000, lo]) {
  const d = await getJson(`poet.song.${v}.json`);
  if (!d) { console.log(`  卷 ${v} 拉取失败`); continue; }
  raw += d.length;
  for (const p of d) {
    if (!Array.isArray(p.paragraphs)) continue;
    for (const f of FORMS) {
      if (p.paragraphs.length !== f.segs) continue;
      if (p.paragraphs.every((s) => f.re.test(s))) { kept++; stats[f.name]++; break; }
    }
  }
  console.log(`  卷 ${v}: ${d.length} 首原始`);
}
console.log(`\n抽样 ${raw} 首 → 合格 ${kept} 首  产出率 ${(100 * kept / raw).toFixed(1)}%`);
console.log(`  体裁分布:`, stats);
const est = Math.round((lo / 1000 + 1) * (kept / 3));
console.log(`\n估算宋诗全量可得 ≈ ${est.toLocaleString()} 首（唐诗现有 35454 首）`);
console.log(`注：以上正则跑在繁体原文上，繁转简后数量不变，仅用于估算`);
