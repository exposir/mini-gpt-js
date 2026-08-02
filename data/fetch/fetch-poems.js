// ============================================================
// fetch-poems.js —— 从开源数据集 chinese-poetry（公版《全唐诗》）
// 自动拉取并筛选标准五言绝句，繁转简后合并进 poems.js
//
// 用法: node fetch-poems.js [目标总数=400]
// 依赖: npm install opencc-js  （仅数据准备用，训练/生成仍零依赖）
// ============================================================

const fs = require("fs");
const path = require("path");
const OpenCC = require("opencc-js");

const TARGET = Number(process.argv[2]) || 400;
const POEMS_FILE = path.join(__dirname, "../poems.js");

// 繁体 → 简体转换器
const toCN = OpenCC.Converter({ from: "t", to: "cn" });

// 数据源镜像（依次尝试）：chinese-poetry 全唐诗分卷 JSON，每卷 1000 首
const MIRRORS = [
  f => `https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/%E5%85%A8%E5%94%90%E8%AF%97/${f}`,
  f => `https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/${f}`,
  f => `https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/json/${f}`,
];

async function fetchJson(file) {
  for (const mk of MIRRORS) {
    const url = mk(file);
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(30000) });
      if (!res.ok) continue;
      return await res.json();
    } catch (e) { /* 换下一个镜像 */ }
  }
  return null;
}

// 标准五绝判定：2 段，每段 "5字，5字。"（转简体后再验，排除缺字 □ 等）
const SEG = /^[\u4e00-\u9fff]{5}，[\u4e00-\u9fff]{5}。$/;
function extractWujue(poems) {
  const out = [];
  for (const p of poems) {
    if (!Array.isArray(p.paragraphs) || p.paragraphs.length !== 2) continue;
    const cn = p.paragraphs.map(s => toCN(s));
    if (cn.every(s => SEG.test(s))) out.push(cn.join(""));
  }
  return out;
}

(async () => {
  // 1. 读取现有语料（用户维护的部分，优先保留）
  const existing = require(POEMS_FILE);
  console.log(`现有语料: ${existing.length} 首`);

  // 2. 逐卷拉取，直到候选足够（多拉一些用于优选）
  const seen = new Set(existing);
  const candidates = [];
  for (let vol = 0; vol < 58000 && candidates.length < (TARGET - existing.length) * 3; vol += 1000) {
    const file = `poet.tang.${vol}.json`;
    const data = await fetchJson(file);
    if (!data) { console.log(`  ${file} 拉取失败，跳过`); continue; }
    const found = extractWujue(data);
    for (const poem of found) {
      if (!seen.has(poem)) { seen.add(poem); candidates.push(poem); }
    }
    console.log(`  ${file}: 五绝 ${found.length} 首（候选累计 ${candidates.length}）`);
  }

  if (existing.length + candidates.length < TARGET) {
    console.log(`候选不足（${existing.length + candidates.length} < ${TARGET}），有多少用多少`);
  }

  // 3. 优选：统计候选池字频，优先选"用字常见"的诗 —— 控制词表规模，利于小模型学习
  const freq = {};
  for (const poem of candidates)
    for (const ch of poem.replace(/[，。]/g, "")) freq[ch] = (freq[ch] || 0) + 1;
  const score = poem => {
    let s = 0;
    for (const ch of poem.replace(/[，。]/g, "")) s += Math.log(freq[ch]);
    return s;
  };
  candidates.sort((a, b) => score(b) - score(a));

  const need = Math.max(0, TARGET - existing.length);
  const merged = [...existing, ...candidates.slice(0, need)];

  // 4. 终检：格式、去重
  const bad = merged.filter(s => s.length !== 24 || !SEG.test(s.slice(0, 12)) || !SEG.test(s.slice(12)));
  if (bad.length) { console.error("格式异常，中止:", bad.slice(0, 3)); process.exit(1); }
  if (new Set(merged).size !== merged.length) { console.error("存在重复，中止"); process.exit(1); }

  // 5. 直接写盘（fs 写入，不经编辑器）
  const body = merged.map(s => `  ${JSON.stringify(s)},`).join("\n");
  fs.writeFileSync(POEMS_FILE,
    `// poems.js —— 训练语料：标准五言绝句（4 句 × 5 字）\n` +
    `// 前 ${existing.length} 首为人工精选；其余来自公版《全唐诗》（chinese-poetry 数据集，繁转简）\n` +
    `module.exports = [\n${body}\n];\n`);

  const vocab = new Set(merged.join("")).size;
  console.log(`\n完成: 共 ${merged.length} 首（新增 ${merged.length - existing.length}），词表 ${vocab} 字`);
  console.log(`已写入 ${POEMS_FILE}`);
})();
