// ============================================================
// gpu-server.js —— 小诗机 GPU 版（Deno + WebGPU 推理）
//
// 服务端: /api/poem 用 GPU 生成（比 CPU 版快 ~15 倍）
// 浏览器: 页面提供「本地模式」——下载权重后在你的浏览器里
//         用同一份 webgpu-forward.js 就地推理，断网也能作诗
//
// 启动: deno run --no-code-cache --allow-read --allow-net gpu-server.js [端口]
// ============================================================

import { createPoet } from "./webgpu-forward.js";
import { chars, stoi } from "./data-split.js";

const PORT = Number(Deno.args[0]) || 8888;

// 两个模型并存，网页上可切。不是为了好玩——两者的差异本身就是结论：
// 验证集曲线证明 17.5 万步是泛化最优点，35 万步已过拐点、记忆探针测出 20% 背诵率。
const MODELS = {
  gen: {
    prefix: "poet-weights-v2-best",
    name: "泛化最优 (17.5 万步)",
    note: "val loss 4.51，记忆探针 0% —— 它在真作，但用词较寡淡",
  },
  mem: {
    prefix: "poet-weights",
    name: "旧版 (35 万步)",
    note: "已过拐点，val loss 5.31，记忆探针 20% —— 词汇丰富但会背原句",
  },
};

for (const m of Object.values(MODELS)) {
  m.meta = JSON.parse(Deno.readTextFileSync(`./${m.prefix}.meta.json`));
  m.bin = Deno.readFileSync(`./${m.prefix}.bin`);
  console.log(`加载 ${m.name}: ${m.meta.cfg.nLayer}层/${m.meta.cfg.nEmbd}维, step ${m.meta.step}`);
  m.poet = await createPoet(m.meta, m.bin.buffer, chars);
}
console.log("GPU 推理就绪（两个模型）");

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
  <div class="sub">零依赖手写 GPT · 5330 万参数 · 35454 首全唐诗<br>训练与推理全程 WebGPU</div>
  <div class="row">
    <input id="start" placeholder="给一个字，或一句诗（如：月 / 故人西辞黄鹤楼）" maxlength="12">
    <button id="go" onclick="gen()">作 诗</button>
  </div>
  <div class="pick" id="pick"></div>
  <div class="mode">
    <label><input type="radio" name="mode" value="server" checked> 服务端 GPU</label>
    <label><input type="radio" name="mode" value="local"> 浏览器本地推理（首次需下载 213MB 权重）</label>
    <span id="localState"></span>
  </div>
  <div id="out"></div>
  <div class="tip">给 1 个字 → 随机创作 | 给一句诗 → 接龙续写（五言/七言自动识别）<br>
    本地模式：模型在你浏览器的 GPU 里运行，断网可用</div>
<script type="module">
import { createPoet } from "/webgpu-forward.js";
const esc = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// 模型选单由服务端列表生成，避开两处硬编的描述对不上
const models = await (await fetch("/models.json")).json();
document.getElementById("pick").innerHTML = Object.entries(models).map(([k, m], i) =>
  '<label><input type="radio" name="model" value="' + k + '"' + (i === 0 ? " checked" : "") + '> ' +
  '<span class="nm">' + esc(m.name) + '</span><br><span class="nt">' + esc(m.note) + '</span></label>').join("");
const curModel = () => document.querySelector('input[name="model"]:checked').value;

const localPoets = {};                 // 每个模型各缓一份，切回来不用重下
async function ensureLocal(key) {
  if (localPoets[key]) return localPoets[key];
  const st = document.getElementById("localState");
  st.textContent = "下载词表...";
  const chars = await (await fetch("/vocab.json")).json();
  st.textContent = "下载权重 213MB...";
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

const FORWARD_SRC = Deno.readTextFileSync("./webgpu-forward.js");

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
    const bad = [...start].filter((c) => !(c in stoi));
    if (bad.length) {
      return Response.json({ error: `「${bad.join("、")}」不在词表中，换个字试试` });
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
    return Response.json(chars);
  }
  if (p === "/models.json") {
    return Response.json(Object.fromEntries(
      Object.entries(MODELS).map(([k, m]) => [k, { name: m.name, note: m.note, step: m.meta.step }]),
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
