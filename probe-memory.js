// probe-memory.js —— 记忆探针
// 拿 704 首留出诗的首句当提示：A(验证最优) 从未见过这些诗，B(线上版) 见过。
// 若 B 能复原原句而 A 不能，就是「背诵」的直接证据。
// 用法: deno run --no-code-cache --allow-read probe-memory.js [前缀A] [前缀B]
import { createPoet } from "./webgpu-forward.js";
import { chars, VAL } from "./data-split.js";

async function load(prefix) {
  const meta = JSON.parse(Deno.readTextFileSync(`./${prefix}.meta.json`));
  const bin = Deno.readFileSync(`./${prefix}.bin`);
  return { poet: await createPoet(meta, bin.buffer, chars), meta };
}
const A = await load(Deno.args[0] || "poet-weights-v2-best");
const B = await load(Deno.args[1] || "poet-weights");
console.log(`A = 验证最优 step ${A.meta.step}（没见过下面这些诗）`);
console.log(`B = 线上版本 step ${B.meta.step}（训练时见过下面这些诗）\n`);

// 逐字比对：生成的第二句与原作第二句有多少字相同
const agree = (gen, orig) => {
  let n = 0;
  for (let i = 0; i < Math.min(gen.length, orig.length); i++) if (gen[i] === orig[i]) n++;
  return orig.length ? n / orig.length : 0;
};

let sumA = 0, sumB = 0, cnt = 0;
for (const poem of VAL.slice(0, 10)) {
  const parts = poem.split("，");
  if (parts.length < 2) continue;
  const head = parts[0];                                 // 首句作提示
  const orig2 = parts[1].split("。")[0];                 // 原作第二句
  // temperature 压到 0.1 逼近贪心解码：能背就一定背出来
  const ga = (await A.poet.generateBatch(head, 1, { temperature: 0.1 }))[0];
  const gb = (await B.poet.generateBatch(head, 1, { temperature: 0.1 }))[0];
  const g2 = (s) => (s.split("，")[1] || "").split("。")[0];
  const ra = agree(g2(ga), orig2), rb = agree(g2(gb), orig2);
  sumA += ra; sumB += rb; cnt++;
  console.log(`提示「${head}」`);
  console.log(`  原作  ${orig2}`);
  console.log(`  A     ${g2(ga)}   逐字吻合 ${(ra * 100).toFixed(0)}%`);
  console.log(`  B     ${g2(gb)}   逐字吻合 ${(rb * 100).toFixed(0)}%`);
}
console.log(`\n━━ 平均逐字吻合率（越高 = 越像在背原诗）`);
console.log(`  A 验证最优 175k: ${(100 * sumA / cnt).toFixed(1)}%   ← 从没见过这些诗，这就是「猜」的基线`);
console.log(`  B 线上版本 350k: ${(100 * sumB / cnt).toFixed(1)}%   ← 见过这些诗`);
