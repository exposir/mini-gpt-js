// ============================================================
// gpu-server.js —— 小诗机 GPU 版（Deno + WebGPU 推理）
//
// 服务端: /api/poem 用 GPU 生成（比 CPU 版快 ~15 倍）
// 浏览器: 页面提供「本地模式」——下载权重后在你的浏览器里
//         用同一份 webgpu-forward.js 就地推理，断网也能作诗
//
// 启动: deno run --no-code-cache --allow-read --allow-net --allow-env gpu/gpu-server.js [端口]
// ============================================================

import { loadModel } from "./load-model.js";

const PORT = Number(Deno.args[0]) || 8888;

// 三个模型并存，网页上可切。不是为了好玩——三者的差异就是两个结论：
//   gen vs small 是语料规模的对照（同为泛化最优点，语料差 6.4 倍）
//   gen vs over 是过拟合的对照（同一炉的拐点与终态）
const MODELS = {
  gen: {
    prefix: "poet-weights-v3-best",
    name: "当前最优 · 唐宋 110 万步",
    note: "1009 万字语料，val loss 4.04，重复率 1.5%（真诗 1.9%）",
  },
  small: {
    prefix: "poet-weights-v2-best",
    name: "小语料对照 · 唐诗 17.5 万步",
    note: "157 万字语料，val loss 4.51 —— 同样是泛化最优点，但用词明显寡淡",
  },
  over: {
    prefix: "poet-weights-v3",
    name: "过拟合标本 · 唐宋 200 万步",
    note: "同一炉跑到头，val 从 4.04 升到 4.41、train 降到 2.82 —— 训练集 loss 好看不代表诗好",
  },
};

for (const m of Object.values(MODELS)) {
  Object.assign(m, await loadModel(m.prefix));    // 字表随模型走的规则在 load-model 里统一处理
  m.stoi = Object.fromEntries(m.chars.map((c, i) => [c, i]));
  console.log(`加载 ${m.name}: ${m.meta.cfg.nLayer}层/${m.meta.cfg.nEmbd}维, 词表 ${m.chars.length}, step ${m.meta.step}`);
}
console.log("GPU 推理就绪（三个模型）");

const PAGE = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>小诗机 · 手写 GPT + WebGPU</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Songti SC", "STSong", serif; background: #f5f1e8;
         min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 16px; }
  h1 { color: #6b4f2e; letter-spacing: 8px; margin-bottom: 8px; }
  .sub { color: #a08c6e; font-size: 13px; margin-bottom: 24px; text-align: center; line-height: 1.8; }
  .row { display: flex; gap: 8px; width: 100%; max-width: 460px; margin-bottom: 12px; }
  input { flex: 1; padding: 12px 16px; font-size: 18px; border: 2px solid #d9c9a8;
          border-radius: 8px; background: #fffdf7; outline: none; font-family: inherit; }
  input:focus { border-color: #b08d57; }
  button { padding: 12px 28px; font-size: 16px; border: none; border-radius: 8px;
           background: #8a6d3b; color: #fff; cursor: pointer; font-family: inherit; letter-spacing: 4px; }
  button:disabled { opacity: .5; cursor: wait; }
  .mode { display: flex; gap: 16px; margin-bottom: 12px; font-size: 13px; color: #8a6d3b; align-items: center; flex-wrap: wrap; justify-content: center; max-width: 460px; }
  .pick { width: 100%; max-width: 460px; margin-bottom: 16px; }
  .pick label { display: block; background: #fffdf7; border: 1px solid #e6d9bd; border-radius: 8px;
                padding: 10px 14px; margin-bottom: 6px; cursor: pointer; font-size: 13px; color: #6b4f2e; }
  .pick label:has(input:checked) { border-color: #8a6d3b; background: #faf5e8; }
  .pick .nm { font-weight: bold; }
  .pick .nt { color: #a08c6e; font-size: 12px; }
  .poem { background: #fffdf7; border: 1px solid #e6d9bd; border-radius: 12px;
          padding: 20px 28px; margin-bottom: 14px; width: 100%; max-width: 460px;
          font-size: 20px; line-height: 1.9; color: #4a3a22; text-align: center;
          box-shadow: 0 2px 8px rgba(140,110,60,.08); animation: fade .5s; }
  @keyframes fade { from { opacity: 0; transform: translateY(8px); } }
  .tip { color: #b5a486; font-size: 12px; margin-top: 20px; text-align: center; line-height: 2; }
</style>
</head>
<body>
  <h1>小 诗 机</h1>
  <div class="sub">零依赖手写 GPT · 5500 万参数 · 22.4 万首唐宋诗（1009 万字）<br>训练与推理全程 WebGPU</div>
  <div class="row">
    <input id="start" placeholder="给一个字，或一句诗（如：月 / 故人西辞黄鹤楼）" maxlength="12">
    <button id="go" onclick="gen()">作 诗</button>
  </div>
  <div class="pick" id="pick"></div>
  <div class="mode">
    <label><input type="radio" name="mode" value="server" checked> 服务端 GPU</label>
    <label><input type="radio" name="mode" value="local"> 浏览器本地推理（首次需下载权重）</label>
    <span id="localState"></span>
  </div>
  <div id="out"></div>
  <div class="tip">给 1 个字 → 随机创作 | 给一句诗 → 接龙续写（五言/七言自动识别）<br>
    推理带重复惩罚 2.0（已出现的字扣减 logits，标点除外）——它把重复率从 8.7% 压到 1.5%<br>
    本地模式：模型在你浏览器的 GPU 里运行，断网可用</div>
<script type="module">
import { createPoet } from "/webgpu-forward.js";
const esc = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// 模型选单由服务端列表生成，避开两处硬编的描述对不上
const models = await (await fetch("/models.json")).json();
document.getElementById("pick").innerHTML = Object.entries(models).map(([k, m], i) =>
  '<label><input type="radio" name="model" value="' + k + '"' + (i === 0 ? " checked" : "") + '> ' +
  '<span class="nm">' + esc(m.name) + '</span><br><span class="nt">' + esc(m.note) +
  '<br>词表 ' + m.vocab + ' 字 · 权重 ' + m.mb + 'MB</span></label>').join("");
const curModel = () => document.querySelector('input[name="model"]:checked').value;

const localPoets = {};                 // 每个模型各缓一份，切回来不用重下
async function ensureLocal(key) {
  if (localPoets[key]) return localPoets[key];
  const st = document.getElementById("localState");
  st.textContent = "下载词表...";
  const chars = await (await fetch("/vocab.json?model=" + key)).json();   // 字表必须跟模型走
  st.textContent = "下载权重 " + models[key].mb + "MB...";
  const meta = await (await fetch("/model.meta.json?model=" + key)).json();
  const bin = await (await fetch("/model.bin?model=" + key)).arrayBuffer();
  st.textContent = "初始化 GPU...";
  localPoets[key] = await createPoet(meta, bin, chars);
  st.textContent = "本地就绪 ✓";
  return localPoets[key];
}
window.gen = async function gen() {
  const btn = document.getElementById("go"), out = document.getElementById("out");
  const start = document.getElementById("start").value.trim();
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const key = curModel();
  btn.disabled = true; btn.textContent = "推理中";
  try {
    let poems;
    if (mode === "local") {
      const p = await ensureLocal(key);
      poems = await p.generateBatch(start || "月", 3);
    } else {
      const res = await fetch("/api/poem?model=" + key + "&start=" + encodeURIComponent(start) + "&count=3");
      const data = await res.json();
      if (data.error) { out.innerHTML = '<div class="poem">' + esc(data.error) + "</div>"; return; }
      poems = data.poems;
    }
    out.innerHTML = poems.map(p => '<div class="poem">' + esc(p).replace(/。/g, "。<br>") + "</div>").join("");
  } catch (e) { out.innerHTML = '<div class="poem">异常：' + esc(String(e.message)) + "</div>"; }
  finally { btn.disabled = false; btn.textContent = "作 诗"; }
};
document.getElementById("start").addEventListener("keydown", e => { if (e.key === "Enter") window.gen(); });
</script>
</body>
</html>`;

// 浏览器本地模式要把推理模块原文发过去；路径跟模块走，不看进程当前目录
const FORWARD_SRC = Deno.readTextFileSync(new URL("./webgpu-forward.js", import.meta.url));

// 全服务只有一份显存缓冲，并发请求必须排队，否则会互相踩激活值
let tail = Promise.resolve();
function enqueue(fn) {
  const next = tail.then(fn, fn);
  tail = next.catch(() => {});
  return next;
}

Deno.serve({ port: PORT, hostname: "0.0.0.0" }, async (req) => {
  const url = new URL(req.url);
  const p = url.pathname;

  if (p === "/api/poem") {
    const start = (url.searchParams.get("start") || "").trim().slice(0, 12) || "月";
    const count = Math.min(Number(url.searchParams.get("count")) || 3, 5);
    const m = MODELS[url.searchParams.get("model")] || MODELS.gen;
    const bad = [...start].filter((c) => !(c in m.stoi));   // 词表按模型走
    if (bad.length) {
      return Response.json({ error: `「${bad.join("、")}」不在该模型的词表中（认识 ${m.chars.length} 个字），换个字试试` });
    }
    const t0 = Date.now();
    const poems = await enqueue(() => m.poet.generateBatch(start, count));
    console.log(`[${new Date().toLocaleTimeString()}] "${start}" x${count} ${m.prefix} ${Date.now() - t0}ms`);
    return Response.json({ poems, model: m.name, step: m.meta.step });
  }
  if (p === "/webgpu-forward.js") {
    return new Response(FORWARD_SRC, { headers: { "content-type": "application/javascript; charset=utf-8" } });
  }
  if (p === "/vocab.json") {
    return Response.json((MODELS[url.searchParams.get("model")] || MODELS.gen).chars);
  }
  if (p === "/models.json") {
    return Response.json(Object.fromEntries(
      Object.entries(MODELS).map(([k, m]) => [k, {
        name: m.name, note: m.note, step: m.meta.step,
        vocab: m.chars.length, mb: Math.round(m.bin.byteLength / 1048576),
      }]),
    ));
  }
  if (p === "/model.meta.json") {
    return Response.json((MODELS[url.searchParams.get("model")] || MODELS.gen).meta);
  }
  if (p === "/model.bin") {
    const m = MODELS[url.searchParams.get("model")] || MODELS.gen;
    return new Response(m.bin, { headers: { "content-type": "application/octet-stream" } });
  }
  return new Response(PAGE, { headers: { "content-type": "text/html; charset=utf-8" } });
});

console.log(`小诗机(GPU版)已启动: http://localhost:${PORT}`);
