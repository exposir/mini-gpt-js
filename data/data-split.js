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
export const SEED = 20260729;      // 固定种子，任何时候重跑都是同一批诗

function loadCorpus(name) {
  const url = new URL(`./${name}`, import.meta.url);
  if (name.endsWith(".txt")) {
    if (!existsSync(url)) throw new Error(`语料文件不存在: ${name}（先跑 node fetch-poems-v3.js）`);
    return readFileSync(url, "utf8").split("\n").filter((s) => s.length > 0);
  }
  return require(`./${name}`);
}

// ---------- 按语料名构造完整切分（带缓存）----------
// 对拍不同语料训出的模型时需要同时拿两套字表与切分：
// v2 用唐诗 6379 字表，v3 用唐+宋 9064 字表，模块级单例担不了这个。
const _cache = new Map();
export function loadSplit(name) {
  if (_cache.has(name)) return _cache.get(name);
  const poems = loadCorpus(name);
  const chars = [...new Set("\n" + poems.join("\n") + "\n")].sort();
  const stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
  const valN = Math.min(1600, Math.max(704, Math.floor(poems.length * 0.02 / 32) * 32));
  const idx = poems.map((_, i) => i);
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
  const valIdx = idx.slice(0, valN), trainIdx = idx.slice(valN);
  const train = trainIdx.map((i) => poems[i]);
  const trainSet = new Set(train);
  const r = {
    name, POEMS: poems, chars, stoi, VAL_N: valN,
    encode: (str) => [...str].map((c) => stoi[c]),
    VAL_IDX: valIdx, TRAIN_IDX: trainIdx,
    VAL: valIdx.map((i) => poems[i]), TRAIN: train,
    isInTrain: (line) => {
      for (const p of trainSet) if (p.includes(line)) return true;
      return false;
    },
  };
  _cache.set(name, r);
  return r;
}

// 默认切分（按 CORPUS_NAME）——以下具名导出保持旧调用方不变
const _def = loadSplit(CORPUS_NAME);
export const POEMS = _def.POEMS;
export const chars = _def.chars;
export const stoi = _def.stoi;
export const encode = _def.encode;
export const VAL_N = _def.VAL_N;
export const VAL_IDX = _def.VAL_IDX;
export const TRAIN_IDX = _def.TRAIN_IDX;
export const VAL = _def.VAL;
export const TRAIN = _def.TRAIN;
export const isInTrain = _def.isInTrain;

// 权重文件对应的语料：新版 meta 自带 corpus 字段，旧版（v1/v2）没有，那时只有唐诗。
export function splitForMeta(meta) {
  return loadSplit(meta.corpus || "poems.js");
}

