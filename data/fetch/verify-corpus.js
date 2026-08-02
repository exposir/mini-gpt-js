// verify-corpus.js —— 核对语料切换后的装载、字表、切分是否正常
import * as D from "../data-split.js";

console.log(`语料: ${D.CORPUS_NAME}`);
console.log(`  ${D.POEMS.length.toLocaleString()} 首  ${D.POEMS.join("").length.toLocaleString()} 字  词表 ${D.chars.length}`);
console.log(`  切分: 训练 ${D.TRAIN.length.toLocaleString()} / 验证 ${D.VAL.length}  (VAL_N=${D.VAL_N})`);
console.log(`  验证集 batch 数: ${D.VAL.length / 32}  (须为整数)`);
console.log(`  交集: ${D.VAL.filter((v) => new Set(D.TRAIN).has(v)).length} (应 0)`);

// 长度合法性：四种体裁 24/32/48/64 字符
const okLens = new Set([24, 32, 48, 64]);
const bad = D.POEMS.filter((s) => !okLens.has(s.length));
console.log(`  长度非法: ${bad.length} 首 (应 0)`);

// 编码往返
const t = D.POEMS[0];
const round = D.encode(t).map((i) => D.chars[i]).join("");
console.log(`  编解码往返一致: ${round === t}`);
console.log(`  encode 无 undefined: ${D.encode(D.POEMS.join("").slice(0, 200000)).every((x) => x !== undefined)}`);

// 参数量影响
const V = D.chars.length, E = 640, L = 10;
const wte = V * E, wpe = 65 * E;
const per = 4 * E * E + 2 * E * 4 * E + 4 * E;
console.log(`\n参数量: wte ${(wte / 1e6).toFixed(2)}M + ${L} 层 ${(L * per / 1e6).toFixed(1)}M ≈ ${((wte + wpe + L * per) / 1e6).toFixed(1)}M`);
console.log(`  (旧词表 6379 时是 53.3M)`);
console.log(`\n样例:`);
for (const p of D.POEMS.slice(0, 2)) console.log(`  ${p}`);
console.log(`验证集样例:`);
for (const p of D.VAL.slice(0, 2)) console.log(`  ${p}`);
