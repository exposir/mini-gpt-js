// test-forward.js —— GPU 推理模块数值对拍（vs CPU forward）
import { createRequire } from "node:module";
import { createPoet } from "./webgpu-forward.js";
const require = createRequire(import.meta.url);

const POEMS = require("./poems.js");
const corpus = "\n" + POEMS.join("\n") + "\n";
const chars = [...new Set(corpus)].sort();

const meta = JSON.parse(Deno.readTextFileSync("./poet-weights.meta.json"));
const bin = Deno.readFileSync("./poet-weights.bin");
const poet = await createPoet(meta, bin.buffer, chars);
console.log("GPU 推理模块就绪");

// 数值对拍：同一上下文的 logits 前 5 名应与 CPU 一致
const cpu = require("./mini-gpt-poet.js");
const cpuModel = new cpu.MiniGPT(cpu.CFG);
cpu.loadWeights(cpuModel);
const ctx = cpu.encode("\n床前明月光，");
const t0 = performance.now();
const gpuLogits = await poet.logitsAt(ctx);
const gpuMs = performance.now() - t0;
const t1 = performance.now();
const cpuOut = cpuModel.forward(ctx);
const cpuMs = performance.now() - t1;
const cpuLast = cpuOut.data[cpuOut.data.length - 1];
let maxDiff = 0;
for (let i = 0; i < gpuLogits.length; i++) maxDiff = Math.max(maxDiff, Math.abs(gpuLogits[i] - cpuLast[i]));
const top5 = (a) => Array.from(a).map((v, i) => i).sort((x, y) => a[y] - a[x]).slice(0, 5).map((i) => chars[i]).join("");
console.log(`logits 最大误差: ${maxDiff.toExponential(2)}`);
console.log(`top5: GPU=[${top5(gpuLogits)}] CPU=[${top5(cpuLast)}]`);
console.log(`单次前向: GPU ${gpuMs.toFixed(0)}ms vs CPU ${cpuMs.toFixed(0)}ms（${(cpuMs / gpuMs).toFixed(1)}x）`);
if (maxDiff > 0.02) { console.error("对拍失败"); Deno.exit(1); }

// 生成整首计时
const t2 = performance.now();
const poem = await poet.generate("月");
console.log(`\n生成: ${poem}`);
console.log(`整首耗时: ${((performance.now() - t2) / 1000).toFixed(2)}s`);
