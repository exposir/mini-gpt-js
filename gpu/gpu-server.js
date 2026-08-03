// ============================================================
// gpu-server.js —— 小诗机 GPU 版（Deno + WebGPU 推理）
//
// 服务端: /api/poem 用 GPU 生成（比 CPU 版快 ~15 倍）
// 浏览器: 页面提供「本地模式」——下载权重后在你的浏览器里
//         用同一份 webgpu-forward.js 就地推理，断网也能作诗
//
// 启动: deno run --no-code-cache --allow-read --allow-net --allow-env gpu/gpu-server.js [端口]
// ============================================================

import { loadModel, readMeta, weightPath } from "./load-model.js";
import { makeChecker, usable, missing } from "../data/normalize.js";

const PORT = Number(Deno.args[0]) || 8888;

// 繁→简单字表：语料是繁转简洗过的，词表里只有简体。不做这层归一化，
// 「举头望明月」这种最常见的繁体输入会整句被拒。表由 data/build-t2s.js 生成。
const T2S = JSON.parse(Deno.readTextFileSync(new URL("../data/t2s-map.json", import.meta.url)));

// 三个模型并存，网页上可切。不是为了好玩——三者的差异就是两个结论：
//   gen vs small 是语料规模的对照（同为泛化最优点，语料差 6.4 倍）
//   gen vs over 是过拟合的对照（同一炉的拐点与终态）
const MODELS = {
  gen: {
    prefix: "poet-weights-v3-best",
    dlPrefix: "poet-weights-v3-best-i8",   // 本地模式下载 int8 版：210MB → 53MB
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
  m.check = makeChecker(T2S, m.stoi);             // 词表每模型不同，校验器也得每模型一个
  // 有 i8 版的模型，本地模式下载 i8（只为传输，加载时反量化，推理质量等价）。
  // 服务端继续用 f32：显存不省，没理由换。只读文件不建 poet，不重复占显存。
  if (m.dlPrefix) m.dl = { meta: readMeta(m.dlPrefix), bin: Deno.readFileSync(weightPath(`${m.dlPrefix}.bin`)) };
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
  /* 逐字体检条：输入时就告诉你哪个字模型不认识，而不是点了作诗才报错 */
  .diag { width: 100%; max-width: 460px; min-height: 22px; margin: -4px 0 12px;
          display: flex; gap: 4px; align-items: center; flex-wrap: wrap; font-size: 13px; color: #a08c6e; }
  .ch { display: inline-flex; align-items: baseline; gap: 1px; padding: 1px 5px; border-radius: 4px;
        background: #efe7d4; color: #6b4f2e; font-size: 16px; }
  .ch.bad { background: #f6dcd6; color: #a4472e; text-decoration: line-through; }
  .ch.conv { background: #e4ecdc; color: #4a6b3a; }
  .ch em { font-size: 11px; font-style: normal; opacity: .7; }
  .diag .msg { margin-left: 4px; }
  .diag button.mini { padding: 3px 10px; font-size: 12px; letter-spacing: 1px; background: #a4472e; }
  select { padding: 5px 10px; font-size: 13px; border: 1px solid #d9c9a8; border-radius: 6px;
           background: #fffdf7; color: #6b4f2e; font-family: inherit; outline: none; cursor: pointer; }
  button.export { margin-top: 12px; padding: 5px 16px; font-size: 12px; letter-spacing: 2px;
                  background: #fffdf7; color: #8a6d3b; border: 1px solid #d9c9a8; border-radius: 6px; }
</style>
</head>
<body>
  <h1>小 诗 机</h1>
  <div class="sub">零依赖手写 GPT · 5500 万参数 · 22.4 万首唐宋诗（1009 万字）<br>训练与推理全程 WebGPU</div>
  <div class="row">
    <input id="start" placeholder="给一个字，或一句诗（如：月 / 故人西辞黄鹤楼）" maxlength="12">
    <button id="go" onclick="gen()">作 诗</button>
  </div>
  <div class="diag" id="diag"></div>
  <div class="pick" id="pick"></div>
  <div class="mode">
    <label><input type="radio" name="mode" value="server" checked> 服务端 GPU</label>
    <label><input type="radio" name="mode" value="local"> 浏览器本地推理（首次需下载权重）</label>
    <span id="localState"></span>
  </div>
  <div class="mode">
    <span>体裁</span>
    <select id="form">
      <option value="">自由（模型自己定）</option>
      <option value="5-4">五言绝句 · 4句×5字</option>
      <option value="7-4">七言绝句 · 4句×7字</option>
      <option value="5-8">五言律诗 · 8句×5字</option>
      <option value="7-8">七言律诗 · 8句×7字</option>
    </select>
  </div>
  <div id="out"></div>
  <div class="tip">给 1 个字 → 随机创作 | 给一句诗 → 接龙续写（五言/七言自动识别）<br>
    繁体输入自动转简（语料是简体，不转的话「風」「雲」这类字会被当成模型不认识的字）<br>
    推理带重复惩罚 2.0（已出现的字扣减 logits，标点除外）——它把重复率从 8.7% 压到 1.5%<br>
    本地模式：模型在你浏览器的 GPU 里运行，断网可用</div>
<script type="module">
import { createPoet } from "/webgpu-forward.js";
import { makeChecker, usable, missing } from "/normalize.js";
const esc = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// 模型选单由服务端列表生成，避开两处硬编的描述对不上
const models = await (await fetch("/models.json")).json();
document.getElementById("pick").innerHTML = Object.entries(models).map(([k, m], i) =>
  '<label><input type="radio" name="model" value="' + k + '"' + (i === 0 ? " checked" : "") + '> ' +
  '<span class="nm">' + esc(m.name) + '</span><br><span class="nt">' + esc(m.note) +
  '<br>词表 ' + m.vocab + ' 字 · 权重 ' + m.mb + 'MB</span></label>').join("");
const curModel = () => document.querySelector('input[name="model"]:checked').value;

// ---------- 输入体检 ----------
// 繁简表与各模型词表都先拉下来，体检全在本地做：
// 本地模式讲好了断网可用，校验不能反过来依赖服务端。
const T2S = await (await fetch("/t2s.json")).json();
const checkers = {};
for (const k of Object.keys(models)) {
  const chars = await (await fetch("/vocab.json?model=" + k)).json();
  const stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
  checkers[k] = { check: makeChecker(T2S, stoi), chars };
}

let dropOK = false;                    // 用户是否已明确同意去掉不认识的字
function refreshDiag() {
  const text = document.getElementById("start").value.trim();
  const diag = document.getElementById("diag");
  if (!text) { diag.innerHTML = ""; dropOK = false; return null; }
  const rep = checkers[curModel()].check(text);
  const bad = missing(rep);
  diag.innerHTML = rep.map(x =>
    '<span class="ch' + (x.ok ? (x.converted ? ' conv' : '') : ' bad') + '">' + esc(x.raw) +
    (x.converted && x.ok ? '<em>→' + esc(x.use) + '</em>' : '') + '</span>').join("") +
    (bad.length
      ? '<span class="msg">← 标红的字在 22.4 万首唐宋诗里没出现过</span>' +
        (usable(rep) ? ' <button class="mini" onclick="dropBad()">去掉它们</button>' : '')
      : rep.some(x => x.converted) ? '<span class="msg">← 繁体已自动转简</span>' : '');
  return rep;
}
window.dropBad = function () {
  const rep = checkers[curModel()].check(document.getElementById("start").value.trim());
  document.getElementById("start").value = usable(rep);
  dropOK = true;
  refreshDiag();
};
document.getElementById("start").addEventListener("input", () => { dropOK = false; refreshDiag(); });
document.getElementById("pick").addEventListener("change", refreshDiag);   // 切模型要重测：词表不同

const localPoets = {};                 // 每个模型各缓一份，切回来不用重下
// 53MB 权重存进浏览器 Cache API，刷新不重下——本地模式卖点是断网可用，
// 每次刷新都重下就和断网无缘了。caches 只在安全上下文（localhost/https）
// 存在，没有就退回普通 fetch，功能不受影响。
const wcache = "caches" in window ? await caches.open("poet-weights-v1") : null;
async function fetchCached(url, onProgress) {
  const hit = wcache && await wcache.match(url);
  if (hit) return hit.arrayBuffer();
  const res = await fetch(url);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onProgress && total) onProgress(got / total);
  }
  const buf = new Uint8Array(got);
  let p = 0;
  for (const c of chunks) { buf.set(c, p); p += c.length; }
  if (wcache) await wcache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
  return buf.buffer;
}
async function ensureLocal(key) {
  if (localPoets[key]) return localPoets[key];
  const st = document.getElementById("localState");
  st.textContent = "下载词表...";
  const chars = await (await fetch("/vocab.json?model=" + key)).json();   // 字表必须跟模型走
  const meta = JSON.parse(new TextDecoder().decode(await fetchCached("/model.meta.json?model=" + key)));
  const bin = await fetchCached("/model.bin?model=" + key, (f) => {
    st.textContent = "下载权重 " + Math.round(f * 100) + "%（" + models[key].mb + "MB，下次刷新从缓存读）";
  });
  st.textContent = "初始化 GPU...";
  localPoets[key] = await createPoet(meta, bin, chars);
  st.textContent = "本地就绪 ✓";
  return localPoets[key];
}
// ---- 竖排诗笺导出：把一首诗画成宣纸风格的竖排图片（从右往左、从上往下）----
function poemToCard(text) {
  const sents = text.split(/[，。]/).map(s => s.trim()).filter(s => s);
  const maxLen = Math.max(...sents.map(s => [...s].length));
  const nCol = sents.length;
  const fs = nCol >= 8 ? 34 : 40;              // 律诗列多，字号收一点
  const gap = 12, lineH = fs + 10, pad = 46;
  const colW = fs + gap;
  const w = pad * 2 + nCol * colW;
  const h = pad * 2 + maxLen * lineH + 40;
  const c = document.createElement("canvas");
  c.width = w * 2; c.height = h * 2;            // 2 倍采样，导出更清晰
  const g = c.getContext("2d");
  g.scale(2, 2);
  g.fillStyle = "#f5f1e8"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#d9c9a8"; g.lineWidth = 2; g.strokeRect(18, 18, w - 36, h - 36);
  g.fillStyle = "#4a3a22"; g.textAlign = "center"; g.textBaseline = "middle";
  g.font = fs + 'px "Songti SC","STSong",serif';
  for (let i = 0; i < nCol; i++) {
    const x = w - pad - i * colW - colW / 2;    // 竖排：第 i 句在第 i 列，从右往左
    const chars = [...sents[i]];
    for (let j = 0; j < chars.length; j++) g.fillText(chars[j], x, pad + 16 + j * lineH + fs / 2);
  }
  g.font = '14px "Songti SC",serif'; g.fillStyle = "#a08c6e"; g.textAlign = "left";
  g.fillText("小诗机 · 手写 GPT", pad, h - 26);
  return c;
}
function exportCard(text) {
  poemToCard(text).toBlob((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = "诗笺.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
}
window.poemToCard = poemToCard;   // 暴露出来便于调试与测试
window.gen = async function gen() {
  const btn = document.getElementById("go"), out = document.getElementById("out");
  const raw = document.getElementById("start").value.trim();
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const key = curModel();
  // 两个模式走同一套归一化，不能“服务端接受、本地拒绝”
  const rep = raw ? checkers[key].check(raw) : null;
  if (rep && missing(rep).length && !dropOK) { refreshDiag(); return; }   // 体检条已经把原因说清了
  const start = rep ? (dropOK ? usable(rep) : rep.map(x => x.use).join("")) : "月";
  // 体裁："5-4" 等 → {per,lines}；空串 = 自由生成
  const fv = document.getElementById("form").value;
  const form = fv ? { per: Number(fv.split("-")[0]), lines: Number(fv.split("-")[1]) } : null;
  btn.disabled = true; btn.textContent = "推理中";
  // 逐字流式：先摆好三个空诗笺，字来一个字填一个；每首重渲全文，
  // 诗总共不到 70 字，重渲成本可忽略。
  out.innerHTML = "";
  const els = [0, 1, 2].map(() => { const d = document.createElement("div"); d.className = "poem"; d.dataset.t = ""; out.appendChild(d); return d; });
  let idx = 0;
  const paint = (el) => { el.innerHTML = esc(el.dataset.t).replace(/。/g, "。<br>"); };
  // 采样只吐新生成的字，用户给的起笔 start 不在其中——每首开写前先把它填进去，
  // 否则每首都会丢掉开头那个字（也正好把每首固定从用户输入起笔）。
  const startPoem = (i) => { idx = i; els[i].dataset.t = start; paint(els[i]); };
  const feed = (ch) => { els[idx].dataset.t += ch; paint(els[idx]); };
  try {
    if (mode === "local") {
      const p = await ensureLocal(key);
      for (let i = 0; i < 3; i++) {
        startPoem(i);
        if (form) await p.generateForm(start, form, { onToken: feed });
        else await p.generateStepwise(start, { onToken: feed });
      }
    } else {
      const res = await fetch("/api/poem?model=" + key + "&start=" + encodeURIComponent(start) + "&count=3&stream=1" + (fv ? "&form=" + fv : ""));
      const rd = res.body.getReader(), dec = new TextDecoder();
      let buf = "", sep;
      while (true) {
        const { done, value } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        while ((sep = buf.indexOf("\\n\\n")) >= 0) {          // SSE 事件以空行分隔
          const ev = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          const m = /event: (\\w+)\\ndata: (.*)/.exec(ev);
          if (!m) continue;
          const data = JSON.parse(m[2]);
          if (m[1] === "poem") startPoem(data);
          else if (m[1] === "ch") feed(data);
          else if (m[1] === "error") throw new Error(data);
        }
      }
    }
    // 成功后给每首挂「导出诗笺」（此时不再逐字重渲，appendChild 不会吃掉已写的字）
    for (const el of els) {
      if (!el.dataset.t) continue;
      const b = document.createElement("button");
      b.className = "export"; b.textContent = "导出诗笺";
      b.onclick = () => exportCard(el.dataset.t);
      el.appendChild(document.createElement("br"));
      el.appendChild(b);
    }
  } catch (e) { out.innerHTML = '<div class="poem">异常：' + esc(String(e.message)) + "</div>"; }
  finally { btn.disabled = false; btn.textContent = "作 诗"; }
};
document.getElementById("start").addEventListener("keydown", e => { if (e.key === "Enter") window.gen(); });
</script>
</body>
</html>`;

// 浏览器本地模式要把推理模块原文发过去；路径跟模块走，不看进程当前目录
const FORWARD_SRC = Deno.readTextFileSync(new URL("./webgpu-forward.js", import.meta.url));
const UNPACK_SRC = Deno.readTextFileSync(new URL("./unpack-weights.js", import.meta.url));   // forward 内部 import 它
const NORMALIZE_SRC = Deno.readTextFileSync(new URL("../data/normalize.js", import.meta.url));

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
    const raw = (url.searchParams.get("start") || "").trim().slice(0, 12) || "月";
    const count = Math.min(Number(url.searchParams.get("count")) || 3, 5);
    const m = MODELS[url.searchParams.get("model")] || MODELS.gen;
    const report = m.check(raw);
    const bad = missing(report);
    // 繁简转换默认就做（保留了用户意图），但丢掉词表外的字是在改写用户输入，
    // 只在前端已把它们标红、用户明确点了继续（drop=1）时才干。
    const drop = url.searchParams.get("drop") === "1";
    const start = drop ? usable(report) : report.map((x) => x.use).join("");
    if (bad.length && !drop) {
      return Response.json({
        error: `「${bad.join("、")}」在 22.4 万首唐宋诗里一次都没出现过，模型没有它们`,
        missing: bad, usable: usable(report),
      });
    }
    if (!start) return Response.json({ error: "去掉不认识的字后什么都不剩了，换个词试试" });
    // stream=1：SSE 逐字推送。三首改串行（逐字采样一次只能一首），
    // 换「看着它写出来」的体验；不要流式的老路径（generateBatch 并行）保留。
    // form="5-4" 等 → 体裁约束生成；缺省自由生成。
    const fp = (url.searchParams.get("form") || "").split("-").map(Number);
    const form = fp.length === 2 && fp[0] && fp[1] ? { per: fp[0], lines: fp[1] } : null;
    if (url.searchParams.get("stream") === "1") {
      const enc = new TextEncoder();
      const body = new ReadableStream({
        async start(controller) {
          const send = (ev, data) => controller.enqueue(enc.encode(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`));
          const t0 = Date.now();
          try {
            await enqueue(async () => {
              for (let i = 0; i < count; i++) {
                send("poem", i);
                const onToken = (ch) => send("ch", ch);
                if (form) await m.poet.generateForm(start, form, { onToken });
                else await m.poet.generateStepwise(start, { onToken });
              }
            });
            send("done", { model: m.name, used: start });
            console.log(`[${new Date().toLocaleTimeString()}] "${raw}"→"${start}" x${count} ${m.prefix} stream ${Date.now() - t0}ms`);
          } catch (e) {
            send("error", String(e.message || e));
          }
          controller.close();
        },
      });
      return new Response(body, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" } });
    }
    const t0 = Date.now();
    const poems = await enqueue(() => m.poet.generateBatch(start, count));
    console.log(`[${new Date().toLocaleTimeString()}] "${raw}"→"${start}" x${count} ${m.prefix} ${Date.now() - t0}ms`);
    return Response.json({ poems, model: m.name, step: m.meta.step, used: start });
  }
  if (p === "/normalize.js") {                     // 页面与服务端用同一份校验逻辑
    return new Response(NORMALIZE_SRC, { headers: { "content-type": "application/javascript; charset=utf-8" } });
  }
  if (p === "/t2s.json") {
    return Response.json(T2S);
  }
  if (p === "/webgpu-forward.js") {
    return new Response(FORWARD_SRC, { headers: { "content-type": "application/javascript; charset=utf-8" } });
  }
  if (p === "/unpack-weights.js") {                // forward 内部 import 它，本地模式同样需要
    return new Response(UNPACK_SRC, { headers: { "content-type": "application/javascript; charset=utf-8" } });
  }
  if (p === "/vocab.json") {
    return Response.json((MODELS[url.searchParams.get("model")] || MODELS.gen).chars);
  }
  if (p === "/models.json") {
    return Response.json(Object.fromEntries(
      Object.entries(MODELS).map(([k, m]) => [k, {
        name: m.name, note: m.note, step: m.meta.step,
        vocab: m.chars.length,
        mb: Math.round((m.dl ? m.dl.bin : m.bin).byteLength / 1048576),   // 下载体积（i8 优先）
      }]),
    ));
  }
  if (p === "/model.meta.json") {
    const m = MODELS[url.searchParams.get("model")] || MODELS.gen;
    return Response.json(m.dl ? m.dl.meta : m.meta);
  }
  if (p === "/model.bin") {
    const m = MODELS[url.searchParams.get("model")] || MODELS.gen;
    const bin = m.dl ? m.dl.bin : m.bin;
    return new Response(bin, { headers: { "content-type": "application/octet-stream" } });
  }
  return new Response(PAGE, { headers: { "content-type": "text/html; charset=utf-8" } });
});

console.log(`小诗机(GPU版)已启动: http://localhost:${PORT}`);
