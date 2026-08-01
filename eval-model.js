// 验收：生成 + 原创性抽检（600k 成品 vs 300k 快照通用）
const m = require("./mini-gpt-poet.js");
const POEMS = require("./poems.js");
const corpus = POEMS.join("|");
const model = new m.MiniGPT(m.CFG);
m.loadWeights(model);
let total = 0, copied = 0;
for (const start of ["月", "山", "故人西辞黄鹤楼", "断桥是否下过雪"]) {
  console.log(`\n【${start}】`);
  for (let i = 0; i < 3; i++) {
    const ctx = m.encode("\n" + start);
    const out = model.generate(ctx, 65 - start.length, { temperature: 0.6, topK: 5, stopId: m.NL, explore: start.length < 2 });
    const poem = m.decode(out).trim();
    // 原创判定：取第一联查语料
    const firstCouplet = poem.split("。")[0];
    const hit = firstCouplet.length > 6 && corpus.includes(firstCouplet);
    total++; if (hit) copied++;
    console.log(`  ${hit ? "❌抄" : "✅原"} ${poem}`);
  }
}
console.log(`\n原创率: ${total - copied}/${total}`);
