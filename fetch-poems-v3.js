// ============================================================
// fetch-poems-v3.js —— 扩语料：全唐诗 + 全宋诗 的标准 五绝/七绝/五律/七律
//
// 为什么要扩：5330 万参数配 157 万字严重过剩，验证曲线显示 17.5 万步就开始
// 背书。模型想学会用僻词只有两条路——在足够多上下文里见过，或把原句背下来。
// 数据量不够时它没得选。宋诗把语料推到约 1080 万字（7 倍）。
//
// 输出 poems-tangsong.txt（一首一行），不动 poems.js——现有权重的字表
// 是从 poems.js 算出来的，覆盖它会让线上模型静默变成乱码。
//
// 用法: node fetch-poems-v3.js
// ============================================================

const fs = require("fs");
const path = require("path");
const OpenCC = require("opencc-js");

const OUT = path.join(__dirname, "poems-tangsong.txt");
const toCN = OpenCC.Converter({ from: "t", to: "cn" });

// 宋诗与唐诗同放在 chinese-poetry 的「全唐诗」目录下（该仓库的历史遗留）
const MIRRORS = [
  (f) => `https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/%E5%85%A8%E5%94%90%E8%AF%97/${f}`,
  (f) => `https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/${f}`,
];

async function fetchJson(file) {
  for (const mk of MIRRORS) {
    for (let retry = 0; retry < 2; retry++) {
      try {
        const res = await fetch(mk(file), { signal: AbortSignal.timeout(30000) });
        if (res.ok) return await res.json();
        if (res.status === 404) return "404";      // 卷不存在，不必重试
      } catch { /* 超时或网络错，重试 */ }
    }
  }
  return null;
}

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
      const cn = p.paragraphs.map((s) => toCN(s));
      if (cn.every((s) => f.re.test(s))) {
        out.push(cn.join(""));
        stats[f.name]++;
        break;
      }
    }
  }
  return out;
}

// 并发拉取（限流 8 路），保持卷序以便输出稳定
async function fetchAll(files, onDone) {
  const results = new Array(files.length);
  let next = 0;
  const worker = async () => {
    while (true) {
      const i = next++;
      if (i >= files.length) return;
      results[i] = await fetchJson(files[i]);
      onDone(files[i], results[i]);
    }
  };
  await Promise.all(Array.from({ length: 8 }, worker));
  return results;
}

(async () => {
  const files = [];
  for (let v = 0; v < 58000; v += 1000) files.push(`poet.tang.${v}.json`);
  for (let v = 0; v <= 254000; v += 1000) files.push(`poet.song.${v}.json`);
  console.log(`准备拉取 ${files.length} 卷（唐 58 + 宋 255）...`);

  let done = 0, failed = 0, missing = 0;
  const t0 = Date.now();
  const raws = await fetchAll(files, (f, r) => {
    done++;
    if (r === null) { failed++; console.log(`  ${f} 失败`); }
    else if (r === "404") missing++;
    if (done % 40 === 0) {
      console.log(`  进度 ${done}/${files.length}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
    }
  });

  const seen = new Set();
  const all = [];
  const stats = { 五绝: 0, 七绝: 0, 五律: 0, 七律: 0 };
  let rawCount = 0, dup = 0;
  for (const data of raws) {
    if (!data || data === "404") continue;
    rawCount += data.length;
    for (const poem of extract(data, stats)) {
      if (seen.has(poem)) { dup++; continue; }
      seen.add(poem); all.push(poem);
    }
  }

  // 终检：每首必须是四种合法长度之一（含标点 24/32/48/64 字符）
  const okLens = new Set([24, 32, 48, 64]);
  const bad = all.filter((s) => !okLens.has(s.length));
  if (bad.length) {
    console.error(`格式异常 ${bad.length} 首，中止:`, bad.slice(0, 3));
    process.exit(1);
  }
  // 一首一行，正文里不含换行，所以按行存是安全的
  if (all.some((s) => s.includes("\n"))) {
    console.error("有诗正文含换行，按行存储会错位，中止");
    process.exit(1);
  }

  fs.writeFileSync(OUT, all.join("\n") + "\n");

  const totalChars = all.join("").length;
  const vocab = new Set(all.join("")).size;
  console.log(`\n完成: ${all.length.toLocaleString()} 首（原始 ${rawCount.toLocaleString()}，去重丢弃 ${dup.toLocaleString()}）`);
  console.log(`  五绝${stats.五绝} 七绝${stats.七绝} 五律${stats.五律} 七律${stats.七律}`);
  console.log(`  总字数 ${totalChars.toLocaleString()}  词表 ${vocab}  失败卷 ${failed}  空卷 ${missing}`);
  console.log(`  写入 ${OUT}（${(fs.statSync(OUT).size / 1048576).toFixed(1)} MB）`);
  console.log(`\n对比旧语料: 35,454 首 / 1,565,000 字 / 词表 6379`);
})();
