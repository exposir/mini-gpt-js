// ============================================================
// gen.js —— 出诗（挑模型、给题目、要几首）
// 用法: deno run --no-code-cache --allow-read --allow-env tools/gen.js "断桥是否下过雪" [几首] [权重前缀]
// ============================================================
import { loadModel } from "../gpu/load-model.js";

const prompt = Deno.args[0] || "月";
const count = Number(Deno.args[1]) || 6;
const prefix = Deno.args[2] || "poet-weights-v3-best";

const { meta, split, chars, poet } = await loadModel(prefix);

const bad = [...prompt].filter((c) => !(c in split.stoi));
if (bad.length) {
  console.error(`「${bad.join("、")}」不在词表中（该模型认识 ${chars.length} 个字）`);
  Deno.exit(1);
}

console.log(`${prefix}  step ${meta.step}  val ${meta.valLoss?.toFixed(4) ?? "?"}  语料 ${split.name}\n`);

const hanzi = (s) => [...s].filter((c) => /[\u4e00-\u9fff]/.test(c));
const poems = await poet.generateBatch(prompt, count);
poems.forEach((p, i) => {
  const h = hanzi(p);
  const rep = 100 * (1 - new Set(h).size / h.length);
  const copied = p.split("。").map((s) => s.trim()).filter((s) => s.length > 4).filter((s) => split.isInTrain(s));
  console.log(`${i + 1}. ${p.replace(/。/g, "。\n   ").trim()}`);
  console.log(`   [重复 ${rep.toFixed(0)}%${copied.length ? `  ⚠️ 抄袭 ${copied.length} 句` : "  原创"}]\n`);
});
