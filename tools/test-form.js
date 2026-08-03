// 验证 generateForm 的体裁约束：字数、句数、标点结构是否符合格律
import { loadModel } from "../gpu/load-model.js";

const { poet } = await loadModel("poet-weights-v3-best");
const forms = [
  { name: "五绝", per: 5, lines: 4 },
  { name: "七绝", per: 7, lines: 4 },
  { name: "五律", per: 5, lines: 8 },
  { name: "七律", per: 7, lines: 8 },
];

for (const f of forms) {
  console.log(`\n===== ${f.name}（每行 ${f.per} 字 × ${f.lines} 行）=====`);
  for (let i = 0; i < 2; i++) {
    const p = await poet.generateForm("月", { per: f.per, lines: f.lines });
    // 拆句检查：以标点断句
    const sentences = p.split(/[，。]/).filter(s => s.length);
    const lens = sentences.map(s => [...s].length);
    const okLen = lens.every(n => n === f.per);
    const okCount = sentences.length === f.lines;
    console.log(`  ${p}`);
    console.log(`     句长 [${lens.join(",")}] ${okLen ? "✓" : "✗"}  句数 ${sentences.length}/${f.lines} ${okCount ? "✓" : "✗"}`);
  }
}
Deno.exit(0);
