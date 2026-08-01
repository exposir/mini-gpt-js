// compare-models.js —— 同题对比两个快照：验证集最优 vs 线上版
// 用法: deno run --no-code-cache --allow-read compare-models.js [前缀A] [前缀B]
import { createPoet } from "./webgpu-forward.js";
import { chars, isInTrain } from "./data-split.js";

async function load(prefix) {
  const meta = JSON.parse(Deno.readTextFileSync(`./${prefix}.meta.json`));
  const bin = Deno.readFileSync(`./${prefix}.bin`);
  return { poet: await createPoet(meta, bin.buffer, chars), meta };
}

const A = await load(Deno.args[0] || "poet-weights-v2-best");   // 验证集最优
const B = await load(Deno.args[1] || "poet-weights");           // 线上版
console.log(`A = 验证最优  step ${A.meta.step}  val ${A.meta.valLoss?.toFixed(4)}`);
console.log(`B = 线上版本  step ${B.meta.step}\n`);

// 原创性判定：整句在训练集里出现过就算「抄的」
function originality(poem) {
  const lines = poem.split("。").map((s) => s.trim()).filter((s) => s.length > 4);
  let copied = 0;
  for (const ln of lines) if (isInTrain(ln)) copied++;
  return { total: lines.length, copied };
}

const PROMPTS = ["断桥是否下过雪", "床前明月光", "月", "故人西辞黄鹤楼", "春风又绿江南岸"];
const stat = { A: { t: 0, c: 0 }, B: { t: 0, c: 0 } };

for (const q of PROMPTS) {
  console.log(`━━ 「${q}」`);
  for (const [tag, m] of [["A", A], ["B", B]]) {
    const ps = await m.poet.generateBatch(q, 2);
    for (const p of ps) {
      const o = originality(p);
      stat[tag].t += o.total; stat[tag].c += o.copied;
      console.log(`  ${tag}  ${p}    ${o.copied ? `⚠️ ${o.copied}/${o.total} 句抄自训练集` : ""}`);
    }
  }
  console.log();
}

console.log("━━ 原创性汇总（句级，命中训练集原句即算抄袭）");
for (const tag of ["A", "B"]) {
  const s = stat[tag];
  console.log(`  ${tag}: ${s.t - s.c}/${s.t} 句原创  (${(100 * (1 - s.c / s.t)).toFixed(1)}%)`);
}
