// ============================================================
// compare-models.js —— 同题对比任意两个快照
//
// 三把尺子：
//   1) 原创性：整句是否出现在「该模型自己的训练集」里（各按各自语料判，
//      因为对 v2 来说宋诗从没见过，拿宋诗判它抄袭没有意义）
//   2) 重复率：一首诗里重复用字的比例——上一炉暴露的「高楼上楼楼」那类卡字
//   3) 用字广度：去重字数 / 总字数
//
// 用法: deno run --no-code-cache --allow-read --allow-env tools/compare-models.js [前缀A] [前缀B]
// ============================================================
import { loadModel } from "../gpu/load-model.js";

const A = await loadModel(Deno.args[0] || "poet-weights-v3-best");
const B = await loadModel(Deno.args[1] || "poet-weights-v2-best");
for (const [tag, m] of [["A", A], ["B", B]]) {
  console.log(`${tag} = ${m.prefix}  step ${m.meta.step}  val ${m.meta.valLoss?.toFixed(4) ?? "?"}` +
    `  语料 ${m.split.name}(${m.split.POEMS.length.toLocaleString()}首/${m.split.chars.length}字)`);
}
console.log();

// 只取汉字，标点不计入重复率与广度
const hanzi = (s) => [...s].filter((c) => /[\u4e00-\u9fff]/.test(c));

function metrics(poem, split) {
  const lines = poem.split("。").map((s) => s.trim()).filter((s) => s.length > 4);
  let copied = 0;
  for (const ln of lines) if (split.isInTrain(ln)) copied++;
  const h = hanzi(poem);
  const uniq = new Set(h).size;
  return {
    lines: lines.length, copied,
    chars: h.length, uniq,
    repeat: h.length ? 1 - uniq / h.length : 0,   // 重复率：越低越好
  };
}

const PROMPTS = ["断桥是否下过雪", "床前明月光", "月", "故人西辞黄鹤楼", "春风又绿江南岸", "孤舟"];
const stat = {
  A: { lines: 0, copied: 0, chars: 0, uniq: 0, n: 0 },
  B: { lines: 0, copied: 0, chars: 0, uniq: 0, n: 0 },
};

for (const q of PROMPTS) {
  console.log(`━━ 「${q}」`);
  for (const [tag, m] of [["A", A], ["B", B]]) {
    for (const p of await m.poet.generateBatch(q, 2)) {
      const x = metrics(p, m.split);
      const s = stat[tag];
      s.lines += x.lines; s.copied += x.copied; s.chars += x.chars; s.uniq += x.uniq; s.n++;
      const flags = [
        x.copied ? `抄${x.copied}/${x.lines}句` : "",
        x.repeat > 0.25 ? `重复${(100 * x.repeat).toFixed(0)}%` : "",
      ].filter(Boolean).join(" ");
      console.log(`  ${tag}  ${p}${flags ? `   ⚠️ ${flags}` : ""}`);
    }
  }
  console.log();
}

console.log("━━ 汇总");
for (const tag of ["A", "B"]) {
  const s = stat[tag];
  console.log(`  ${tag}  原创 ${s.lines - s.copied}/${s.lines} 句 (${(100 * (1 - s.copied / s.lines)).toFixed(1)}%)` +
    `   重复率 ${(100 * (1 - s.uniq / s.chars)).toFixed(1)}%` +
    `   平均每首用字 ${(s.chars / s.n).toFixed(0)} 个中 ${(s.uniq / s.n).toFixed(1)} 个不同`);
}
console.log(`\n  重复率参考：真唐诗约 ${(() => {
  const real = A.split.POEMS.slice(0, 2000);
  let c = 0, u = 0;
  for (const p of real) { const h = hanzi(p); c += h.length; u += new Set(h).size; }
  return (100 * (1 - u / c)).toFixed(1);
})()}%（取 2000 首真诗实测）`);
