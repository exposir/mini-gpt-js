// ============================================================
// tune-penalty.js —— 扫重复惩罚强度，看重复率与结构完整性
// 用法: deno run --no-code-cache --allow-read --allow-env tools/tune-penalty.js [权重前缀]
// ============================================================
import { loadModel } from "../gpu/load-model.js";

const prefix = Deno.args[0] || "poet-weights-v3-best";
const { meta, split, poet } = await loadModel(prefix);
console.log(`${prefix}  step ${meta.step}  语料 ${split.name}\n`);

const hanzi = (s) => [...s].filter((c) => /[\u4e00-\u9fff]/.test(c));
// 结构合法：整首为 24/32/48/64 字符（含标点），即标准五七言绝律
const OK_LEN = new Set([24, 32, 48, 64]);

const PROMPTS = ["断桥是否下过雪", "床前明月光", "月", "故人西辞黄鹤楼", "孤舟", "春风又绿江南岸", "长安", "雨"];

// 真诗基线
const real = split.POEMS.slice(0, 3000);
let rc = 0, ru = 0;
for (const p of real) { const h = hanzi(p); rc += h.length; ru += new Set(h).size; }
const realRep = 100 * (1 - ru / rc);

console.log("惩罚   重复率   结构合法   抄袭   样例");
console.log("─".repeat(78));
const results = [];
for (const pen of [0, 0.6, 1.2, 2.0, 3.0, 5.0]) {
  let chars = 0, uniq = 0, ok = 0, tot = 0, copied = 0;
  let sample = "";
  for (const q of PROMPTS) {
    const ps = await poet.generateBatch(q, 4, { repPenalty: pen });
    for (const p of ps) {
      const h = hanzi(p);
      chars += h.length; uniq += new Set(h).size; tot++;
      if (OK_LEN.has(p.length)) ok++;
      for (const ln of p.split("。").map((s) => s.trim()).filter((s) => s.length > 4)) {
        if (split.isInTrain(ln)) copied++;
      }
    }
    if (q === "断桥是否下过雪") sample = ps[0];
  }
  const rep = 100 * (1 - uniq / chars);
  results.push({ pen, rep, ok: 100 * ok / tot, copied });
  console.log(`${String(pen).padStart(4)}   ${rep.toFixed(1).padStart(5)}%   ${(100 * ok / tot).toFixed(0).padStart(5)}%     ${String(copied).padStart(3)}   ${sample.slice(0, 32)}`);
}
console.log("─".repeat(78));
console.log(`真诗基线重复率 ${realRep.toFixed(1)}%（3000 首实测）\n`);

console.log("═══ 各强度下同题三首（人工看是否被惩罚逼出怪字）═══");
for (const pen of [0, 1.2, 3.0]) {
  console.log(`\n── 惩罚 ${pen}`);
  for (const p of await poet.generateBatch("断桥是否下过雪", 3, { repPenalty: pen })) {
    const h = hanzi(p);
    console.log(`  ${p}   [重复 ${(100 * (1 - new Set(h).size / h.length)).toFixed(0)}%]`);
  }
}
