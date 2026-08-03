// ============================================================
// build-t2s.js —— 生成繁→简单字映射表（一次性构建，产出 t2s-map.json）
//
// 为什么要预生成而不是运行时调 opencc：
// 浏览器本地模式要能离线跑，不能依赖服务端；而 opencc-js 是 npm 包，
// 项目没有打包器也不想为此引一个。所以在这里把 CJK 基本区扫一遍，
// 把「转换后与原字不同」的单字对固化成一张小表，两端共用。
//
// 语料是用 opencc 繁转简清洗过的，词表里只有简体。不做这层归一化，
// 「舉頭望明月」这种最常见的繁体输入会整句被拒——这不是提示问题，是 bug。
//
// 用法: node data/build-t2s.js
// ============================================================
const oc = require("opencc-js");
const fs = require("fs");
const path = require("path");

const t2s = oc.Converter({ from: "tw", to: "cn" });
const OUT = path.join(__dirname, "t2s-map.json");

const map = {};
let scanned = 0;
for (let cp = 0x4e00; cp <= 0x9fff; cp++) {
  const c = String.fromCodePoint(cp);
  scanned++;
  const s = t2s(c);
  // 只收 1:1 且真的变了的：多字输出会破坏「一个字一个 token」的前提
  if (s.length === 1 && s !== c) map[c] = s;
}

// 港台异体的另一路（tw 与 hk 的取字有差异，合并覆盖更全）
const hk2s = oc.Converter({ from: "hk", to: "cn" });
let extra = 0;
for (let cp = 0x4e00; cp <= 0x9fff; cp++) {
  const c = String.fromCodePoint(cp);
  if (map[c]) continue;
  const s = hk2s(c);
  if (s.length === 1 && s !== c) { map[c] = s; extra++; }
}

fs.writeFileSync(OUT, JSON.stringify(map));
const kb = (fs.statSync(OUT).size / 1024).toFixed(0);
console.log(`扫描 ${scanned.toLocaleString()} 个 CJK 字`);
console.log(`繁简映射 ${Object.keys(map).length.toLocaleString()} 对（其中 ${extra} 对来自港台异体）`);
console.log(`写入 ${path.basename(OUT)}  ${kb}KB`);
