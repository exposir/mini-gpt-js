// 验证 generateForm 的体裁约束：字数、句数、标点结构、押韵是否达标
import { loadModel, RHYME } from "../gpu/load-model.js";

const { poet } = await loadModel("poet-weights-v3-best");
const forms = [
  { name: "五绝", per: 5, lines: 4 },
  { name: "七绝", per: 7, lines: 4 },
  { name: "五律", per: 5, lines: 8 },
  { name: "七律", per: 7, lines: 8 },
];

// 押韵判定：偶数句尾字，第 2 个起都得在锚（第 1 个韵脚）的通押邻居里
const checkRhyme = (poem) => {
  const feet = poem.split("。").filter(g => g).map(g => { const p = g.split("，"); return [...p[p.length - 1]].pop(); });
  const anchor = feet[0];
  const bad = feet.slice(1).filter(f => f !== anchor && !(RHYME[anchor] || "").includes(f));
  return { feet, bad };
};

let fail = 0;
for (const f of forms) {
  console.log(`\n===== ${f.name}（每行 ${f.per} 字 × ${f.lines} 行）=====`);
  for (let i = 0; i < 2; i++) {
    const p = await poet.generateForm("月", { per: f.per, lines: f.lines });
    const sentences = p.split(/[，。]/).filter(s => s.length);
    const lens = sentences.map(s => [...s].length);
    const okLen = lens.every(n => n === f.per);
    const okCount = sentences.length === f.lines;
    const { feet, bad } = checkRhyme(p);
    const okRhyme = bad.length === 0;
    if (!okLen || !okCount || !okRhyme) fail++;
    console.log(`  ${p}`);
    console.log(`     句长 [${lens.join(",")}] ${okLen ? "✓" : "✗"}  句数 ${sentences.length}/${f.lines} ${okCount ? "✓" : "✗"}  韵脚 [${feet.join(",")}] ${okRhyme ? "✓" : "✗ 出韵:" + bad.join(",")}`);
  }
}
console.log(fail === 0 ? "\n全部通过 ✓" : `\n${fail} 首未达标 ✗`);
Deno.exit(fail === 0 ? 0 : 1);
