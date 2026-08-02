// ============================================================
// load-model.js —— 按前缀加载一个权重快照
//
// 这段逻辑原先在 gen / compare-models / probe-memory / score-model /
// tune-penalty 各抄一份。抽出来有两个原因：
//   1) 权重目录只在这里出现一次，搬目录时不用改五个文件
//   2) 「字表随模型走」的规则只在这里实现一次 —— 新版权重把字表存在
//      meta.vocab 里，旧版（v1/v2）没有这个字段，得从它对应的语料现算。
//      抄五份的话，任何一份忘了这条都会静默输出乱码。
//
// 路径用 import.meta.url 而不是 "./x"：Deno.readFileSync 认的是进程
// 当前目录，脚本在 tools/ 下却从仓库根目录启动时，"./" 会指错地方。
// ============================================================
import { createPoet } from "./webgpu-forward.js";
import { splitForMeta } from "../data/data-split.js";

export const weightPath = (file) => new URL(`../weights/${file}`, import.meta.url);

export function readMeta(prefix) {
  return JSON.parse(Deno.readTextFileSync(weightPath(`${prefix}.meta.json`)));
}

export async function loadModel(prefix) {
  const meta = readMeta(prefix);
  const bin = Deno.readFileSync(weightPath(`${prefix}.bin`));
  const split = splitForMeta(meta);              // 该权重对应的语料/字表/切分
  const chars = meta.vocab || split.chars;       // 新版权重自带字表，旧版从语料算
  return { prefix, meta, split, chars, bin, poet: await createPoet(meta, bin.buffer, chars) };
}
