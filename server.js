// ============================================================
// server.js —— 古诗生成器 Web 服务（零依赖，Node 内置 http）
//
// 用法:  node server.js [端口=8888]
// 页面:  http://localhost:8888
// API:   GET /api/poem?start=月&count=3
//
// 给他人使用：同一网络下访问 http://<你的IP>:8888
// ============================================================

const http = require("http");
const fs = require("fs");
const { MiniGPT, CFG, encode, decode, stoi, NL, loadWeights, WEIGHTS } = require("./mini-gpt-poet.js");

const PORT = Number(process.argv[2]) || 8888;

// 启动时加载一次权重（支持二进制 meta+bin 与旧版 JSON 两种格式）
const hasBin = fs.existsSync(WEIGHTS.replace(/\.json$/, ".meta.json")) && fs.existsSync(WEIGHTS.replace(/\.json$/, ".bin"));
if (!hasBin && !fs.existsSync(WEIGHTS)) {
  console.error("未找到权重文件，请先训练: node mini-gpt-poet.js --train");
  process.exit(1);
}
const model = new MiniGPT(CFG);
const meta = loadWeights(model);
console.log(`权重已加载 (训练进度 ${meta.step || "?"}/${meta.totalSteps || "?"} 步)`);

function generatePoems(start, count) {
  const poems = [];
  for (let i = 0; i < count; i++) {
    const ctx = encode("\n" + start);
    const out = model.generate(ctx, 65 - start.length,
      { temperature: 0.6, topK: 5, stopId: NL, explore: start.length < 2 });
    poems.push(decode(out).trim());
  }
  return poems;
}

const PAGE = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>小诗机 · 纯 JS 手写 GPT</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Songti SC", "STSong", serif; background: #f5f1e8;
         min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 16px; }
  h1 { color: #6b4f2e; letter-spacing: 8px; margin-bottom: 8px; }
  .sub { color: #a08c6e; font-size: 13px; margin-bottom: 32px; }
  .row { display: flex; gap: 8px; width: 100%; max-width: 420px; margin-bottom: 24px; }
  input { flex: 1; padding: 12px 16px; font-size: 18px; border: 2px solid #d9c9a8;
          border-radius: 8px; background: #fffdf7; outline: none; font-family: inherit; }
  input:focus { border-color: #b08d57; }
  button { padding: 12px 28px; font-size: 16px; border: none; border-radius: 8px;
           background: #8a6d3b; color: #fff; cursor: pointer; font-family: inherit; letter-spacing: 4px; }
  button:disabled { opacity: .5; cursor: wait; }
  .poem { background: #fffdf7; border: 1px solid #e6d9bd; border-radius: 12px;
          padding: 20px 28px; margin-bottom: 14px; width: 100%; max-width: 420px;
          font-size: 20px; line-height: 1.9; color: #4a3a22; text-align: center;
          box-shadow: 0 2px 8px rgba(140,110,60,.08); animation: fade .5s; }
  @keyframes fade { from { opacity: 0; transform: translateY(8px); } }
  .tip { color: #b5a486; font-size: 12px; margin-top: 20px; text-align: center; line-height: 2; }
</style>
</head>
<body>
  <h1>小 诗 机</h1>
  <div class="sub">纯 JavaScript 手写 Transformer · 三千七百首唐诗训练</div>
  <div class="row">
    <input id="start" placeholder="给一个字，或一句诗（如：月 / 床前明月光）" maxlength="12">
    <button id="go" onclick="gen()">作 诗</button>
  </div>
  <div id="out"></div>
  <div class="tip">给 1 个字 → 随机创作 &nbsp;|&nbsp; 给一句诗 → 接龙续写<br>
    模型只认识唐诗里出现过的字</div>
<script>
const esc = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function gen() {
  const btn = document.getElementById("go"), out = document.getElementById("out");
  const start = document.getElementById("start").value.trim();
  btn.disabled = true; btn.textContent = "推理中";
  try {
    const res = await fetch("/api/poem?start=" + encodeURIComponent(start) + "&count=3");
    const data = await res.json();
    if (data.error) { out.innerHTML = '<div class="poem">' + esc(data.error) + "</div>"; return; }
    out.innerHTML = data.poems.map(p => '<div class="poem">' + esc(p).replace(/。/g, "。<br>") + "</div>").join("");
  } catch (e) { out.innerHTML = '<div class="poem">服务异常：' + esc(String(e.message)) + "</div>"; }
  finally { btn.disabled = false; btn.textContent = "作 诗"; }
}
document.getElementById("start").addEventListener("keydown", e => { if (e.key === "Enter") gen(); });
</script>
</body>
</html>`;

http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (url.pathname === "/api/poem") {
    const start = (url.searchParams.get("start") || "").trim().slice(0, 12);
    const count = Math.min(Number(url.searchParams.get("count")) || 3, 5);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    const bad = [...start].filter(c => !(c in stoi));
    if (bad.length) {
      res.end(JSON.stringify({ error: `「${bad.join("、")}」不在词表中，换个字试试（唐诗里出现过的字都可以）` }));
      return;
    }
    const t0 = Date.now();
    const poems = generatePoems(start, count);
    console.log(`[${new Date().toLocaleTimeString()}] start="${start}" count=${count} ${Date.now() - t0}ms`);
    res.end(JSON.stringify({ poems }));
    return;
  }

  res.setHeader("Content-Type", "text/html; charset=utf-8");
  res.end(PAGE);
}).listen(PORT, "0.0.0.0", () => {
  console.log(`小诗机已启动:  http://localhost:${PORT}`);
  // 打印局域网地址，方便分享给同事
  const nets = require("os").networkInterfaces();
  for (const list of Object.values(nets))
    for (const n of list)
      if (n.family === "IPv4" && !n.internal)
        console.log(`局域网访问:    http://${n.address}:${PORT}`);
});
