// ============================================================
// probe-memory.js —— 记忆探针
//
// 拿「该模型自己的留出诗」首句当提示，temperature 压到 0.1 逼近贪心解码：
// 能背就一定背出来。逐字吻合率高 = 在复现原作，低 = 在真作。
//
// 每个模型用各自语料的留出集（v2 是 704 首唐诗，v3 是 1600 首唐+宋），
// 两者都是「该模型训练时从未见过」的诗，这才是可比的口径。
//
// 用法: deno run --no-code-cache --allow-read --allow-env tools/probe-memory.js [前缀A] [前缀B]
// ============================================================
import { loadModel } from "../gpu/load-model.js";

const A = await loadModel(Deno.args[0] || "poet-weights-v3-best");
const B = await loadModel(Deno.args[1] || "poet-weights-v3");
const sameSet = A.split.name === B.split.name;
for (const [tag, m] of [["A", A], ["B", B]]) {
  console.log(`${tag} = ${m.prefix}  step ${m.meta.step}  val ${m.meta.valLoss?.toFixed(4) ?? "?"}  留出集 ${m.split.name}(${m.split.VAL.length}首)`);
}
console.log(sameSet ? "两者留出集相同，逐题可直接对比\n" : "两者语料不同，各用自己的留出集（口径可比，题目不同）\n");

// 逐字比对：生成的第二句与原作第二句有多少字位置相同
const agree = (gen, orig) => {
  let n = 0;
  for (let i = 0; i < Math.min(gen.length, orig.length); i++) if (gen[i] === orig[i]) n++;
  return orig.length ? n / orig.length : 0;
};
const second = (s) => (s.split("，")[1] || "").split("。")[0];

const N = 12;
const sum = { A: 0, B: 0 };
const hits = { A: 0, B: 0 };          // 吻合率 > 50% 记为一次「背出来」

for (let i = 0; i < N; i++) {
  const rows = [];
  for (const [tag, m] of [["A", A], ["B", B]]) {
    const poem = m.split.VAL[i];
    const parts = poem.split("，");
    if (parts.length < 2) continue;
    const head = parts[0], orig2 = parts[1].split("。")[0];
    const gen = (await m.poet.generateBatch(head, 1, { temperature: 0.1 }))[0];
    const r = agree(second(gen), orig2);
    sum[tag] += r;
    if (r > 0.5) hits[tag]++;
    rows.push({ tag, head, orig2, gen2: second(gen), r });
  }
  if (sameSet) {
    console.log(`提示「${rows[0].head}」  原作: ${rows[0].orig2}`);
    for (const x of rows) console.log(`  ${x.tag}  ${x.gen2}   吻合 ${(x.r * 100).toFixed(0)}%${x.r > 0.5 ? "  ⚠️ 在背原作" : ""}`);
  } else {
    for (const x of rows) {
      console.log(`  ${x.tag} 提示「${x.head}」 原作 ${x.orig2} → 生成 ${x.gen2}   吻合 ${(x.r * 100).toFixed(0)}%${x.r > 0.5 ? "  ⚠️" : ""}`);
    }
  }
}

console.log(`\n━━ 平均逐字吻合率（越高 = 越像在背原诗）`);
for (const [tag, m] of [["A", A], ["B", B]]) {
  console.log(`  ${tag} ${m.prefix} (step ${m.meta.step}): ${(100 * sum[tag] / N).toFixed(1)}%   其中 ${hits[tag]}/${N} 首吻合过半`);
}
