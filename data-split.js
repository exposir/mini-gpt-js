// ============================================================
// data-split.js —— 语料、词表、训练/验证切分的唯一来源
//
// 为什么要单独一个文件：切分逻辑曾在 gpu-train.js / compare-models.js /
// probe-memory.js 各抄一份。任何一处手抄错一个字符，验证集就不是同一批诗，
// 而后果是静默的——不报错，只给出错的结论。这类 bug 必须靠单一来源根除。
//
// 词表也放这里：字表决定权重矩阵的行数，各处必须逐字一致，否则权重对不上。
// ============================================================

import { createRequire } from "node:module";
import { readFileSync, existsSync } from "node:fs";
const require = createRequire(import.meta.url);

// ---------- 语料选择 ----------
// 默认仍是 poems.js（唐诗 35454 首），现有权重的字表就是从它算出来的，
// 默认不变才能保证线上模型不被惊动。换大语料用环境变量显式指定：
//   POET_CORPUS=poems-tangsong.txt deno run ... gpu-train.js
// .txt 按行读（一首一行），.js 走 require。
// 读环境变量包 try：Deno 没给 --allow-env 时 Deno.env.get 会抛，
// 不能因此让所有旧启动命令（如 gpu-server）都要加权限。
function envCorpus() {
  try {
    if (typeof Deno !== "undefined") return Deno.env.get("POET_CORPUS");
    return process.env.POET_CORPUS;
  } catch { return undefined; }
}
export const CORPUS_NAME = envCorpus() || "poems.js";

function loadCorpus(name) {
  const url = new URL(`./${name}`, import.meta.url);
  if (name.endsWith(".txt")) {
    if (!existsSync(url)) throw new Error(`语料文件不存在: ${name}（先跑 node fetch-poems-v3.js）`);
    return readFileSync(url, "utf8").split("\n").filter((s) => s.length > 0);
  }
  return require(`./${name}`);
}

export const POEMS = loadCorpus(CORPUS_NAME);

// ---------- 词表 ----------
const corpus = "\n" + POEMS.join("\n") + "\n";
export const chars = [...new Set(corpus)].sort();
export const stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
export const encode = (s) => [...s].map((c) => stoi[c]);

// ---------- 训练/验证切分 ----------
// 验证集取 2%，但上下都卡住并对齐 batch size 32：
//   下限 704（唐诗语料正好是 704，保持与 v2 那一炉的曲线可比）
//   上限 1600（=50 个 batch）——再多也不会更准，只是拖慢评估：
//   语料涨到 24 万首时 2% 是 4896 首，一次评估要 153 个 batch、约 115 秒。
export const VAL_N = Math.min(1600, Math.max(704, Math.floor(POEMS.length * 0.02 / 32) * 32));
export const SEED = 20260729;      // 固定种子，任何时候重跑都是同一批诗

const shuffled = (() => {
  const idx = POEMS.map((_, i) => i);
  let s = SEED;
  const rnd = () => {               // mulberry32：Math.imul 保证 32 位不丢精度
    s |= 0; s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
})();

export const VAL_IDX = shuffled.slice(0, VAL_N);      // 留出，全程不参与训练
export const TRAIN_IDX = shuffled.slice(VAL_N);
export const VAL = VAL_IDX.map((i) => POEMS[i]);
export const TRAIN = TRAIN_IDX.map((i) => POEMS[i]);

// 判断某句是否抄自训练集（原创性抽检用）
const TRAIN_SET = new Set(TRAIN);
export function isInTrain(line) {
  for (const p of TRAIN_SET) if (p.includes(line)) return true;
  return false;
}
