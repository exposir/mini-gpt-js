// test-kv.js —— KV Cache 正确性对拍 + 速度对比
import { createRequire } from "node:module";
import { createPoet } from "../gpu/webgpu-forward.js";
const require = createRequire(import.meta.url);

const POEMS = require("../data/poems.js");
const chars = [...new Set("\n" + POEMS.join("\n") + "\n")].sort();
const stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
const enc = (s) => [...s].map((c) => stoi[c]);

const meta = JSON.parse(Deno.readTextFileSync(new URL("../weights/poet-weights.meta.json", import.meta.url)));
const bin = Deno.readFileSync(new URL("../weights/poet-weights.bin", import.meta.url));
const poet = await createPoet(meta, bin.buffer, chars);
console.log("GPU 就绪\n");

// ---- 1) 数值对拍：KV Cache 路径 vs 全序列路径，同一上下文 logits 应一致 ----
console.log("― 数值对拍（KV Cache vs 全序列重算）―");
let allOk = true;
for (const text of ["\n月", "\n床前明月光，", "\n故人西辞黄鹤楼，烟花三月下扬州。"]) {
  const ids = enc(text);
  const a = await poet.logitsKV(ids);
  const b = await poet.logitsAt(ids);
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  const top = (x) => Array.from(x).map((v, i) => i).sort((p, q) => x[q] - x[p]).slice(0, 5).map((i) => chars[i]).join("");
  const ok = maxDiff < 5e-3 && top(a) === top(b);
  allOk = allOk && ok;
  console.log(`  len=${ids.length}: 最大误差 ${maxDiff.toExponential(2)}  top5 KV=[${top(a)}] 全序列=[${top(b)}] ${ok ? "✅" : "❌"}`);
}
if (!allOk) { console.error("对拍失败，中止"); Deno.exit(1); }

// ---- 2) 公平基准：固定工作量，测「每 token 成本」 ----
console.log("\n― 单 token 成本（上下文已 40 字，各测 10 次取均值）―");
const ctx40 = enc("\n" + "月落江边一鴈飞，远寻遗迹久依依。凭君休向朝朝去，".repeat(2)).slice(0, 40);

// A: 全序列重算（每字都把 65 行序列算一遍）
let t = performance.now();
for (let i = 0; i < 10; i++) await poet.logitsAt(ctx40);
const msFull = (performance.now() - t) / 10;

// B: KV Cache 增量（cache 已填好，只算新增 1 行）
await poet.logitsKV(ctx40);                       // 先填满 cache
t = performance.now();
for (let i = 0; i < 10; i++) await poet.stepOne(40);
const msKV = (performance.now() - t) / 10;

// C: 分段提交（12 个 token 串在一个 pass，摊到每 token）
t = performance.now();
for (let i = 0; i < 5; i++) await poet.stepChunk(40, 52);
const msChunk = (performance.now() - t) / 5 / 12;

console.log(`  A 全序列重算:      ${msFull.toFixed(1)} ms/token`);
console.log(`  B KV Cache 逐字:   ${msKV.toFixed(1)} ms/token   (${(msFull / msKV).toFixed(1)}x)`);
console.log(`  C KV+分段一次提交: ${msChunk.toFixed(1)} ms/token   (${(msFull / msChunk).toFixed(1)}x)`);

// ---- 3) 批处理：出 N 首 vs 出 1 首 ----
console.log("\n― 批处理（同一题目生成 N 首，强制 40 字）―");
for (const n of [1, 2, 4, 8]) {
  const t0 = performance.now();
  const ps = await poet.generateBatch("故人西辞黄鹤楼", n, { maxNew: 40 });
  const ms = performance.now() - t0;
  console.log(`  ${n} 首: ${(ms / 1000).toFixed(2)}s  (每首 ${(ms / n / 1000).toFixed(2)}s)  样例: ${ps[0].slice(8, 24)}...`);
}

// ---- 4) 生成质量与稳定性 ----
console.log("\n― 一次批量出 3 首（验证批内互不干扰）―");
for (const p of await poet.generateBatch("断桥是否下过雪", 3)) console.log(`  ${p}`);
console.log("\n― 五言（提前收尾）―");
for (const p of await poet.generateBatch("床前明月光", 2)) console.log(`  ${p}`);
