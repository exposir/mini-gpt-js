// ============================================================
// build-site.js —— 同步静态站点 docs/（GitHub Pages 用）
//
// docs/ 是「公网单模型版」：纯浏览器 WebGPU 推理，没有服务端。
// 页面（index.html）是静态版专属、手工维护；但算法与数据绝不手拷——
// forward/unpack/normalize/韵表/繁简表/权重都以 gpu|data|weights 为
// 单一来源，改了那边之后跑一次本脚本再提交。
//
// 用法: deno run --allow-read --allow-write tools/build-site.js
// ============================================================
const root = (p) => new URL(`../${p}`, import.meta.url);

const FILES = [
  ["gpu/webgpu-forward.js", "docs/webgpu-forward.js"],
  ["gpu/unpack-weights.js", "docs/unpack-weights.js"],
  ["data/normalize.js", "docs/normalize.js"],
  ["data/t2s-map.json", "docs/t2s.json"],
  ["data/rhyme-map.json", "docs/rhyme.json"],
  ["weights/poet-weights-v3-best-i8.meta.json", "docs/model.meta.json"],
  ["weights/poet-weights-v3-best-i8.bin", "docs/model.bin"],
];

Deno.mkdirSync(root("docs"), { recursive: true });
let total = 0;
for (const [src, dst] of FILES) {
  const data = Deno.readFileSync(root(src));
  Deno.writeFileSync(root(dst), data);
  total += data.length;
  console.log(`${src.padEnd(45)} → ${dst}  ${(data.length / 1048576).toFixed(1)}MB`);
}
console.log(`\n共 ${(total / 1048576).toFixed(1)}MB（不含 index.html）`);
