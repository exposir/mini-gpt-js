// ============================================================
// fetch-poems-v2.js —— 阶段3：从 chinese-poetry（公版《全唐诗》）
// 拉取全部标准 五绝/七绝/五律/七律，繁转简，重建 poems.js
//
// 用法: node fetch-poems-v2.js
// 依赖: opencc-js（仅数据准备用，训练/生成仍零依赖）
// ============================================================

const fs = require("fs");
const path = require("path");
const OpenCC = require("opencc-js");

const POEMS_FILE = path.join(__dirname, "poems.js");
const toCN = OpenCC.Converter({ from: "t", to: "cn" });

const MIRRORS = [
  f => `https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/%E5%85%A8%E5%94%90%E8%AF%97/${f}`,
  f => `https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/${f}`,
];

async function fetchJson(file) {
  for (const mk of MIRRORS) {
    try {
      const res = await fetch(mk(file), { signal: AbortSignal.timeout(30000) });
      if (!res.ok) continue;
      return await res.json();
    } catch { /* 换镜像 */ }
  }
  return null;
}

// 四种标准体裁：段数 × 每段格式
const SEG5 = /^[\u4e00-\u9fff]{5}，[\u4e00-\u9fff]{5}。$/;
const SEG7 = /^[\u4e00-\u9fff]{7}，[\u4e00-\u9fff]{7}。$/;
const FORMS = [
  { name: "五绝", segs: 2, re: SEG5 },
  { name: "七绝", segs: 2, re: SEG7 },
  { name: "五律", segs: 4, re: SEG5 },
  { name: "七律", segs: 4, re: SEG7 },
];

function extract(poems, stats) {
  const out = [];
  for (const p of poems) {
    if (!Array.isArray(p.paragraphs)) continue;
    for (const f of FORMS) {
      if (p.paragraphs.length !== f.segs) continue;
      const cn = p.paragraphs.map(s => toCN(s));
      if (cn.every(s => f.re.test(s))) {
        out.push(cn.join(""));
        stats[f.name]++;
        break;
      }
    }
  }
  return out;
}

(async () => {
  const seen = new Set();
  const all = [];
  const stats = { 五绝: 0, 七绝: 0, 五律: 0, 七律: 0 };
  let failed = 0;

  for (let vol = 0; vol < 58000; vol += 1000) {
    const file = `poet.tang.${vol}.json`;
    const data = await fetchJson(file);
    if (!data) { failed++; console.log(`  ${file} 失败跳过`); continue; }
    for (const poem of extract(data, stats)) {
      if (!seen.has(poem)) { seen.add(poem); all.push(poem); }
    }
    if (vol % 10000 === 0) console.log(`  进度 ${vol / 1000}/58 卷，累计 ${all.length} 首`);
  }

  // 终检：每首必须是四种合法长度之一（24/32/48/64 字符含标点）
  const okLens = new Set([24, 32, 48, 64]);
  const bad = all.filter(s => !okLens.has(s.length));
  if (bad.length) { console.error("格式异常，中止:", bad.slice(0, 3)); process.exit(1); }

  const body = all.map(s => `  ${JSON.stringify(s)},`).join("\n");
  fs.writeFileSync(POEMS_FILE,
    `// poems.js —— 训练语料：全唐诗标准 五绝/七绝/五律/七律（chinese-poetry 数据集，繁转简）\n` +
    `module.exports = [\n${body}\n];\n`);

  const totalChars = all.join("").length;
  const vocab = new Set(all.join("")).size;
  console.log(`\n完成: ${all.length} 首 | 五绝${stats.五绝} 七绝${stats.七绝} 五律${stats.五律} 七律${stats.七律}`);
  console.log(`总字数 ${totalChars.toLocaleString()} | 词表 ${vocab} | 失败卷 ${failed}`);
})();
