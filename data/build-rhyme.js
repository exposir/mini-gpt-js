// ============================================================
// build-rhyme.js —— 从语料自举押韵表（零外部依赖）
//
// 为什么不用平水韵书：一来外部数据源在本机网络不可达；二来更本质——
// 模型是从这份语料学的，mask 也应该来自同一份语料。韵书是「应然」，
// 语料共现是唐宋诗人的「实然」（含支微、鱼虞这类实际通押），
// 用实然做约束，mask 与模型的倾向一致，不会把模型逼进它不熟的角落。
//
// 方法：
//   1) 只取结构规整的近体诗：恰 4/8 句、每句同为 5/7 字——韵脚最可靠
//   2) 每首诗取所有「。」前的字为韵脚组（首句尾字可韵可不韵，不采）
//   3) 组内两两计共现。不聚类成韵部：多音字（如「斜」）会让并查集
//      把两个韵部塌缩成一个。直接存邻接表：allow(字) = 与它同押 ≥N 次的字
//   4) N=3 滤掉抄写错误与古体通押的长尾噪声
//
// 用法: deno run --allow-read --allow-write data/build-rhyme.js
// ============================================================
const MIN_CO = 3;                    // 共现 ≥3 次才算「真的通押」

const src = new URL("./poems-tangsong.txt", import.meta.url);
const lines = Deno.readTextFileSync(src).split("\n");

// 结构判定：4 或 8 个「X{n}，X{n}。」段，n 恒为 5 或 7
const isRegular = (sents) => {
  if (sents.length !== 4 && sents.length !== 8) return false;
  const n = [...sents[0]].length;
  if (n !== 5 && n !== 7) return false;
  return sents.every((s) => [...s].length === n);
};

const pairs = new Map();             // "a|b" (a<b) → 共现次数
const bump = (a, b) => {
  const k = a < b ? a + "|" + b : b + "|" + a;
  pairs.set(k, (pairs.get(k) || 0) + 1);
};

let used = 0;
for (const line of lines) {
  const t = line.trim();
  if (!t) continue;
  // 拆成句子：以。分组，每组内可能有一个，
  const groups = t.split("。").filter((g) => g);
  const sents = groups.flatMap((g) => g.split("，")).filter((s) => s);
  if (!isRegular(sents)) continue;
  // 韵脚 = 每个偶数句的尾字（即。前一字）
  const rhymes = groups.map((g) => { const p = g.split("，"); const last = p[p.length - 1]; return [...last].pop(); }).filter(Boolean);
  if (rhymes.length < 2) continue;
  used++;
  for (let i = 0; i < rhymes.length; i++) {
    for (let j = i + 1; j < rhymes.length; j++) {
      if (rhymes[i] !== rhymes[j]) bump(rhymes[i], rhymes[j]);
    }
  }
}

// 邻接表：只保留强边
const allow = new Map();             // 字 → Set<字>
for (const [k, n] of pairs) {
  if (n < MIN_CO) continue;
  const [a, b] = k.split("|");
  if (!allow.has(a)) allow.set(a, new Set());
  if (!allow.has(b)) allow.set(b, new Set());
  allow.get(a).add(b);
  allow.get(b).add(a);
}

// 值存成字符串（一字一 char），比数组省一半体积
const out = {};
for (const [ch, set] of [...allow.entries()].sort()) out[ch] = [...set].sort().join("");
const dst = new URL("./rhyme-map.json", import.meta.url);
Deno.writeTextFileSync(dst, JSON.stringify(out));

const sizes = [...allow.values()].map((s) => s.size).sort((x, y) => x - y);
const mid = sizes[Math.floor(sizes.length / 2)];
console.log(`格律诗 ${used.toLocaleString()} 首（语料 ${lines.length.toLocaleString()} 行）`);
console.log(`韵脚字 ${allow.size}，邻居数 中位 ${mid} / 最大 ${sizes[sizes.length - 1]}`);
console.log(`共现对 ${pairs.size.toLocaleString()}，强边(≥${MIN_CO}) ${[...pairs.values()].filter((n) => n >= MIN_CO).length.toLocaleString()}`);
console.log(`落盘 ${(Deno.statSync(dst).size / 1024).toFixed(0)}KB → ${dst.pathname}`);
// sanity：东 应含 中/风/空，不含 江/山
for (const probe of ["东", "时", "家"]) {
  const s = out[probe] || "";
  console.log(`  ${probe} → ${s.slice(0, 30)}${s.length > 30 ? `...(共${[...s].length})` : ""}`);
}
