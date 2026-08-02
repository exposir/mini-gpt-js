// ============================================================
// score-model.js —— 给单个快照打分，输出一行便于横比
//
// 平台区几个候选的 val 只差 0.03，分辨不了。换三把尺子：
//   固有重复   关掉惩罚时的重复率 —— 模型自己有多想卡字
//   用字广度   开惩罚（=上线配置）时跨 48 首用到多少不同的字，
//              「寡淡」的直接度量：只会循环用高频字的模型广度低
//   结构合法   整首落在 24/32/48/64 字符（标准五七言绝律）的比例
//
// 用法: deno run --no-code-cache --allow-read --allow-env tools/score-model.js 前缀
// ============================================================
import { loadModel } from "../gpu/load-model.js";

const prefix = Deno.args[0];
const { meta, split, poet } = await loadModel(prefix);

const hanzi = (s) => [...s].filter((c) => /[\u4e00-\u9fff]/.test(c));
const OK_LEN = new Set([24, 32, 48, 64]);
const PROMPTS = ["断桥是否下过雪", "床前明月光", "月", "故人西辞黄鹤楼", "孤舟", "春风又绿江南岸", "长安", "雨"];

async function run(pen) {
  const seen = new Set();
  let chars = 0, uniqIn = 0, ok = 0, tot = 0, copied = 0;
  for (const q of PROMPTS) {
    for (const p of await poet.generateBatch(q, 6, { repPenalty: pen })) {
      const h = hanzi(p);
      chars += h.length; uniqIn += new Set(h).size; tot++;
      for (const c of h) seen.add(c);
      if (OK_LEN.has(p.length)) ok++;
      for (const ln of p.split("。").map((s) => s.trim()).filter((s) => s.length > 4)) {
        if (split.isInTrain(ln)) copied++;
      }
    }
  }
  return { rep: 100 * (1 - uniqIn / chars), breadth: seen.size, chars, ok: 100 * ok / tot, copied };
}

const off = await run(0);       // 固有倾向
const on = await run(2.0);      // 上线配置
console.log([
  prefix,
  meta.step,
  (meta.valLoss ?? NaN).toFixed(4),
  off.rep.toFixed(1),
  on.breadth,
  (1000 * on.breadth / on.chars).toFixed(0),
  on.ok.toFixed(0),
  on.copied,
].join("\t"));
