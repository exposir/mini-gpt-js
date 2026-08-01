// ============================================================
// mini-gpt-poet.js —— 能实际使用的小功能：古诗灵感生成器
// 基于纯 JS 手写 Transformer（autograd + Adam），零依赖
//
// 用法：
//   node mini-gpt-poet.js            # 首次自动训练并存盘权重，之后秒开
//   node mini-gpt-poet.js 月         # 以"月"开头生成诗句
//   node mini-gpt-poet.js 月 5       # 生成 5 首
//   node mini-gpt-poet.js --train    # 强制重新训练
// ============================================================

const fs = require("fs");
const path = require("path");
const WEIGHTS = path.join(__dirname, "poet-weights.json");

// WASM SIMD 矩阵乘核（编译/自检失败会自动回退纯 JS）
const wasmKernel = require("./wasm-kernel.js");

// ---------- 1. 基础数学 ----------

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Float32Array(cols));
}

function randn(rows, cols, std = 0.05) {
  const m = zeros(rows, cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const u = 1 - Math.random(), v = Math.random();
      m[i][j] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
    }
  return m;
}

function softmaxRow(row) {
  const max = Math.max(...row);
  const exps = row.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

const GC = Math.sqrt(2 / Math.PI), GA = 0.044715;
const gelu = x => 0.5 * x * (1 + Math.tanh(GC * (x + GA * x ** 3)));
function geluGrad(x) {
  const t = Math.tanh(GC * (x + GA * x ** 3));
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GC * (1 + 3 * GA * x * x);
}

// ---------- 2. 微型 autograd ----------

class Tensor {
  constructor(data) {
    this.data = data;
    this.grad = data.map(r => new Float32Array(r.length));
    this._backward = () => {};
    this._parents = [];
  }
}

function backward(loss) {
  const topo = [], visited = new Set();
  (function build(t) {
    if (visited.has(t)) return;
    visited.add(t);
    for (const p of t._parents) build(p);
    topo.push(t);
  })(loss);
  loss.grad[0][0] = 1;
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
}

// ---------- 3. 算子（前向 + 手推反向） ----------

function matmul(A, B) {
  const n = A.data.length, k = B.data.length, m = B.data[0].length;
  const C = new Tensor(zeros(n, m));
  if (wasmKernel.USE_WASM && m % 4 === 0) {
    const R = wasmKernel.matmul(A.data, B.data);   // 前向走 WASM SIMD
    for (let i = 0; i < n; i++) C.data[i] = R[i];
  } else {
    for (let i = 0; i < n; i++)
      for (let p = 0; p < k; p++) {
        const a = A.data[i][p];
        for (let j = 0; j < m; j++) C.data[i][j] += a * B.data[p][j];
      }
  }
  C._parents = [A, B];
  C._backward = () => {
    if (wasmKernel.USE_WASM && k % 4 === 0 && m % 4 === 0) {
      // dA(n×k) = C.grad(n×m) · Bᵀ(m×k)，累加回 A.grad
      const dA = wasmKernel.matmul(C.grad, wasmKernel.transpose(B.data));
      for (let i = 0; i < n; i++) { const g = A.grad[i], d = dA[i]; for (let p = 0; p < k; p++) g[p] += d[p]; }
      // dB(k×m) = Aᵀ(k×n) · C.grad(n×m)，累加回 B.grad
      const dB = wasmKernel.matmul(wasmKernel.transpose(A.data), C.grad);
      for (let p = 0; p < k; p++) { const g = B.grad[p], d = dB[p]; for (let j = 0; j < m; j++) g[j] += d[j]; }
    } else {
      for (let i = 0; i < n; i++)
        for (let p = 0; p < k; p++) {
          let s = 0;
          for (let j = 0; j < m; j++) {
            s += C.grad[i][j] * B.data[p][j];
            B.grad[p][j] += A.data[i][p] * C.grad[i][j];
          }
          A.grad[i][p] += s;
        }
    }
  };
  return C;
}

function matmulT(A, W) {
  const n = A.data.length, d = A.data[0].length, V = W.data.length;
  const C = new Tensor(zeros(n, V));
  for (let i = 0; i < n; i++)
    for (let v = 0; v < V; v++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += A.data[i][j] * W.data[v][j];
      C.data[i][v] = s;
    }
  C._parents = [A, W];
  C._backward = () => {
    for (let i = 0; i < n; i++)
      for (let v = 0; v < V; v++) {
        const g = C.grad[i][v];
        if (g === 0) continue;
        for (let j = 0; j < d; j++) {
          A.grad[i][j] += g * W.data[v][j];
          W.grad[v][j] += g * A.data[i][j];
        }
      }
  };
  return C;
}

function add(A, B) {
  const n = A.data.length, m = A.data[0].length;
  const C = new Tensor(zeros(n, m));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) C.data[i][j] = A.data[i][j] + B.data[i][j];
  C._parents = [A, B];
  C._backward = () => {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) {
        A.grad[i][j] += C.grad[i][j];
        B.grad[i][j] += C.grad[i][j];
      }
  };
  return C;
}

function geluOp(X) {
  const n = X.data.length, m = X.data[0].length;
  const C = new Tensor(zeros(n, m));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) C.data[i][j] = gelu(X.data[i][j]);
  C._parents = [X];
  C._backward = () => {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++)
        X.grad[i][j] += geluGrad(X.data[i][j]) * C.grad[i][j];
  };
  return C;
}

function layerNorm(X, g, b) {
  const n = X.data.length, d = X.data[0].length;
  const C = new Tensor(zeros(n, d));
  const xhat = zeros(n, d), invstd = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let mean = 0;
    for (const v of X.data[i]) mean += v / d;
    let variance = 0;
    for (const v of X.data[i]) variance += (v - mean) ** 2 / d;
    invstd[i] = 1 / Math.sqrt(variance + 1e-5);
    for (let j = 0; j < d; j++) {
      xhat[i][j] = (X.data[i][j] - mean) * invstd[i];
      C.data[i][j] = xhat[i][j] * g.data[0][j] + b.data[0][j];
    }
  }
  C._parents = [X, g, b];
  C._backward = () => {
    for (let i = 0; i < n; i++) {
      let m1 = 0, m2 = 0;
      const dyhat = new Float32Array(d);
      for (let j = 0; j < d; j++) {
        dyhat[j] = C.grad[i][j] * g.data[0][j];
        m1 += dyhat[j] / d;
        m2 += dyhat[j] * xhat[i][j] / d;
        g.grad[0][j] += C.grad[i][j] * xhat[i][j];
        b.grad[0][j] += C.grad[i][j];
      }
      for (let j = 0; j < d; j++)
        X.grad[i][j] += invstd[i] * (dyhat[j] - m1 - xhat[i][j] * m2);
    }
  };
  return C;
}

function causalAttention(Q, K, V, nHead) {
  const T = Q.data.length, nEmbd = Q.data[0].length, hd = nEmbd / nHead;
  const C = new Tensor(zeros(T, nEmbd));
  const attnW = [];
  for (let h = 0; h < nHead; h++) {
    const off = h * hd; attnW[h] = [];
    for (let i = 0; i < T; i++) {
      const scores = new Array(i + 1);
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        for (let d = 0; d < hd; d++) dot += Q.data[i][off + d] * K.data[j][off + d];
        scores[j] = dot / Math.sqrt(hd);
      }
      const w = softmaxRow(scores);
      attnW[h][i] = w;
      for (let j = 0; j <= i; j++)
        for (let d = 0; d < hd; d++) C.data[i][off + d] += w[j] * V.data[j][off + d];
    }
  }
  C._parents = [Q, K, V];
  C._backward = () => {
    for (let h = 0; h < nHead; h++) {
      const off = h * hd;
      for (let i = 0; i < T; i++) {
        const w = attnW[h][i];
        const dA = new Float32Array(i + 1);
        for (let j = 0; j <= i; j++)
          for (let d = 0; d < hd; d++) {
            dA[j] += C.grad[i][off + d] * V.data[j][off + d];
            V.grad[j][off + d] += w[j] * C.grad[i][off + d];
          }
        let sum = 0;
        for (let j = 0; j <= i; j++) sum += w[j] * dA[j];
        for (let j = 0; j <= i; j++) {
          const dS = w[j] * (dA[j] - sum) / Math.sqrt(hd);
          for (let d = 0; d < hd; d++) {
            Q.grad[i][off + d] += dS * K.data[j][off + d];
            K.grad[j][off + d] += dS * Q.data[i][off + d];
          }
        }
      }
    }
  };
  return C;
}

function embed(wte, wpe, ids) {
  const T = ids.length, d = wte.data[0].length;
  const C = new Tensor(zeros(T, d));
  for (let i = 0; i < T; i++)
    for (let j = 0; j < d; j++) C.data[i][j] = wte.data[ids[i]][j] + wpe.data[i][j];
  C._parents = [wte, wpe];
  C._backward = () => {
    for (let i = 0; i < T; i++)
      for (let j = 0; j < d; j++) {
        wte.grad[ids[i]][j] += C.grad[i][j];
        wpe.grad[i][j] += C.grad[i][j];
      }
  };
  return C;
}

function crossEntropy(logits, targets) {
  const T = logits.data.length;
  const probs = logits.data.map(row => softmaxRow([...row]));
  let sum = 0;
  for (let i = 0; i < T; i++) sum += -Math.log(probs[i][targets[i]] + 1e-12);
  const C = new Tensor([new Float32Array([sum / T])]);
  C._parents = [logits];
  C._backward = () => {
    const g = C.grad[0][0] / T;
    for (let i = 0; i < T; i++)
      for (let j = 0; j < probs[i].length; j++)
        logits.grad[i][j] += (probs[i][j] - (j === targets[i] ? 1 : 0)) * g;
  };
  return C;
}

// ---------- 4. GPT 模型 ----------

const ones = d => new Tensor([new Float32Array(d).fill(1)]);
const zerosP = d => new Tensor([new Float32Array(d)]);

class MiniGPT {
  constructor({ vocabSize, blockSize, nLayer, nHead, nEmbd }) {
    this.nHead = nHead;
    this.blockSize = blockSize;
    this.wte = new Tensor(randn(vocabSize, nEmbd));
    this.wpe = new Tensor(randn(blockSize, nEmbd));
    this.blocks = Array.from({ length: nLayer }, () => ({
      ln1g: ones(nEmbd), ln1b: zerosP(nEmbd),
      Wq: new Tensor(randn(nEmbd, nEmbd)), Wk: new Tensor(randn(nEmbd, nEmbd)),
      Wv: new Tensor(randn(nEmbd, nEmbd)), Wo: new Tensor(randn(nEmbd, nEmbd)),
      ln2g: ones(nEmbd), ln2b: zerosP(nEmbd),
      W1: new Tensor(randn(nEmbd, 4 * nEmbd)), W2: new Tensor(randn(4 * nEmbd, nEmbd)),
    }));
    this.lnFg = ones(nEmbd); this.lnFb = zerosP(nEmbd);
  }

  params() {
    const ps = [this.wte, this.wpe, this.lnFg, this.lnFb];
    for (const b of this.blocks)
      ps.push(b.ln1g, b.ln1b, b.Wq, b.Wk, b.Wv, b.Wo, b.ln2g, b.ln2b, b.W1, b.W2);
    return ps;
  }

  forward(ids) {
    let X = embed(this.wte, this.wpe, ids);
    for (const b of this.blocks) {
      const h = layerNorm(X, b.ln1g, b.ln1b);
      const attnOut = causalAttention(
        matmul(h, b.Wq), matmul(h, b.Wk), matmul(h, b.Wv), this.nHead);
      X = add(X, matmul(attnOut, b.Wo));
      const h2 = layerNorm(X, b.ln2g, b.ln2b);
      X = add(X, matmul(geluOp(matmul(h2, b.W1)), b.W2));
    }
    X = layerNorm(X, this.lnFg, this.lnFb);
    return matmulT(X, this.wte);
  }

  // 采样策略：开头几字高温度（逐不同的起句），后面低温度（稳定续写）
  generate(ids, maxNewTokens, { temperature = 0.75, topK = 8, stopId = -1, explore = true } = {}) {
    ids = [...ids];
    for (let s = 0; s < maxNewTokens; s++) {
      const logits = this.forward(ids.slice(-this.blockSize));
      const last = [...logits.data[logits.data.length - 1]];
      let next;
      if (explore && s < 2) {
        // 开头 2 字：在 top-4 候选里均匀随机选，强制多样化起句
        const idx = last.map((v, i) => i).sort((a, b) => last[b] - last[a]).slice(0, 4);
        next = idx[Math.floor(Math.random() * idx.length)];
      } else {
        const idx = last.map((v, i) => i).sort((a, b) => last[b] - last[a]).slice(0, topK);
        const probs = softmaxRow(idx.map(i => last[i] / temperature));
        let r = Math.random(); next = idx[0];
        for (let i = 0; i < idx.length; i++) { r -= probs[i]; if (r <= 0) { next = idx[i]; break; } }
      }
      if (next === stopId) break;
      ids.push(next);
    }
    return ids;
  }
}

// ---------- 5. Adam 优化器 ----------

class Adam {
  constructor(params, lr = 3e-3) {
    this.params = params; this.lr = lr; this.t = 0;
    this.m = params.map(p => p.data.map(r => new Float32Array(r.length)));
    this.v = params.map(p => p.data.map(r => new Float32Array(r.length)));
  }
  zeroGrad() { for (const p of this.params) for (const row of p.grad) row.fill(0); }
  step() {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    this.params.forEach((p, k) => {
      for (let i = 0; i < p.data.length; i++)
        for (let j = 0; j < p.data[i].length; j++) {
          const g = p.grad[i][j];
          this.m[k][i][j] = b1 * this.m[k][i][j] + (1 - b1) * g;
          this.v[k][i][j] = b2 * this.v[k][i][j] + (1 - b2) * g * g;
          p.data[i][j] -= this.lr * (this.m[k][i][j] / c1) /
                          (Math.sqrt(this.v[k][i][j] / c2) + eps);
        }
    });
  }
}

// ---------- 6. 训练语料：经典五言绝句（公版），见 poems.js ----------

const POEMS = require("./poems.js");

// 每首诗前后加换行符作为"开始/结束"标记
const corpus = "\n" + POEMS.join("\n") + "\n";
const chars = [...new Set(corpus)].sort();
const stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
const itos = Object.fromEntries(chars.map((c, i) => [i, c]));
const encode = s => [...s].map(c => stoi[c]);
const decode = ids => ids.map(i => itos[i]).join("");
const NL = stoi["\n"];

const CFG = { vocabSize: chars.length, blockSize: 66, nLayer: 10, nHead: 10, nEmbd: 640 };

// ---------- 7. 权重存取（训练一次，之后秒开） ----------

function saveWeights(model, step = 0, totalSteps = 0) {
  const payload = {
    cfg: { vocabSize: CFG.vocabSize, nLayer: CFG.nLayer, nHead: CFG.nHead, nEmbd: CFG.nEmbd },
    step, totalSteps,
    data: model.params().map(p => p.data.map(r => Array.from(r))),
  };
  const tmp = WEIGHTS + ".tmp";                    // 原子写：先写临时文件再替换，
  fs.writeFileSync(tmp, JSON.stringify(payload));  // 中途崩溃不会损坏旧权重
  fs.renameSync(tmp, WEIGHTS);
}

function loadWeights(model) {
  // 优先二进制格式（meta.json + bin，GPU 训练器产出）；JSON 在 5000 万参数下超 V8 字符串上限
  const metaPath = WEIGHTS.replace(/\.json$/, ".meta.json");
  const binPath = WEIGHTS.replace(/\.json$/, ".bin");
  if (fs.existsSync(metaPath) && fs.existsSync(binPath)) {
    const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
    const c = meta.cfg;
    if (!c || c.vocabSize !== CFG.vocabSize || c.nLayer !== CFG.nLayer || c.nEmbd !== CFG.nEmbd) {
      console.error("二进制权重与当前模型结构/词表不匹配");
      process.exit(1);
    }
    const raw = fs.readFileSync(binPath);
    let off = 0;
    model.params().forEach((p, k) => {
      const { rows, cols } = meta.tensors[k];
      p.data = [];
      for (let i = 0; i < rows; i++) {
        p.data.push(new Float32Array(raw.buffer, raw.byteOffset + off, cols).slice());
        off += cols * 4;
      }
      p.grad = p.data.map(r => new Float32Array(r.length));
    });
    return meta;
  }
  const payload = JSON.parse(fs.readFileSync(WEIGHTS, "utf8"));
  const c = payload.cfg;
  if (!c || c.vocabSize !== CFG.vocabSize || c.nLayer !== CFG.nLayer || c.nEmbd !== CFG.nEmbd) {
    console.error("存盘权重与当前模型结构/词表不匹配，请先 --train 重新训练");
    process.exit(1);
  }
  model.params().forEach((p, k) => {
    p.data = payload.data[k].map(r => Float32Array.from(r));
    p.grad = p.data.map(r => new Float32Array(r.length));
  });
  return payload;
}

// ---------- 8. 训练：每步随机抽一首诗（首尾带 \n 边界标记） ----------

function train(model, totalSteps = 200000, lr = 3e-3, startStep = 0, accum = 1) {
  console.log(`语料: ${POEMS.length} 首五言绝句, ${corpus.length} 字, 词表 ${chars.length}`);
  console.log(`训练 ${startStep + 1} → ${totalSteps} 步 (lr=${lr}, 梯度累积 batch=${accum})...`);
  const opt = new Adam(model.params(), lr);
  const t0 = Date.now();
  let lossSum = 0, lossN = 0;
  for (let step = startStep + 1; step <= totalSteps; step++) {
    // 学习率衰减：后期调小步长，让 loss 稳定收敛（真实大模型训练的标配）
    opt.lr = step > totalSteps * 0.6 ? lr / 3 : lr;
    const poem = POEMS[Math.floor(Math.random() * POEMS.length)];
    const ids = encode("\n" + poem + "\n");        // \n诗\n，学会"从头写到尾"
    const loss = crossEntropy(model.forward(ids.slice(0, -1)), ids.slice(1));
    backward(loss);                                // 梯度累加进 p.grad（天然支持累积）
    if (step % accum === 0) {                      // 每 accum 首诗才更新一次：
      opt.step();                                  // 梯度噪声 ↓accum 倍，方向更稳
      opt.zeroGrad();
    }
    lossSum += loss.data[0][0]; lossN++;
    if (step % 5000 === 0) {
      const speed = ((Date.now() - t0) / 1000 / (step - startStep)).toFixed(3);
      console.log(`  step ${step}  平均loss = ${(lossSum / lossN).toFixed(4)}  (${speed}s/步)`);
      lossSum = 0; lossN = 0;
    }
    if (step % 10000 === 0) {
      saveWeights(model, step, totalSteps);        // 周期 checkpoint，断点可续
      console.log(`  [checkpoint] 已存盘 @ step ${step}`);
    }
  }
  saveWeights(model, totalSteps, totalSteps);
  console.log(`训练完成，耗时 ${((Date.now() - t0) / 3600000).toFixed(2)} 小时，权重已存盘\n`);
}

// ---------- 9. 模块导出 & 命令行入口 ----------

module.exports = { MiniGPT, CFG, encode, decode, stoi, chars, NL,
  loadWeights, saveWeights, WEIGHTS, backward, crossEntropy };

if (require.main === module) {

const args = process.argv.slice(2);
const forceTrain = args.includes("--train");
const resume = args.includes("--resume");
const more = args.includes("--more");
const rest = args.filter(a => a !== "--train" && a !== "--resume" && a !== "--more");
const startChar = rest.find(a => isNaN(Number(a))) || "";
const count = Number(rest.find(a => !isNaN(Number(a)))) || 3;

const model = new MiniGPT(CFG);

if (more && fs.existsSync(WEIGHTS)) {
  // 增量精炼：在现有权重上用梯度累积(batch=8)+低学习率继续训练
  const meta = loadWeights(model);
  const from = meta.step || 0;
  console.log(`增量精炼：从 step ${from} 再训 200000 样本（batch=8）...`);
  train(model, from + 200000, 1e-3, from, 8);
} else if (resume && fs.existsSync(WEIGHTS)) {
  // 断点续训：从 checkpoint 记录的步数接着练到目标步数
  const meta = loadWeights(model);
  if ((meta.step || 0) >= (meta.totalSteps || 0)) {
    console.log("上次训练已跑完，直接使用；如需重练请用 --train\n");
  } else {
    console.log(`断点续训：从 step ${meta.step} 继续到 ${meta.totalSteps}...`);
    train(model, meta.totalSteps, 3e-3, meta.step);
  }
} else if (!forceTrain && fs.existsSync(WEIGHTS)) {
  loadWeights(model);
  console.log("已加载存盘权重（免训练）\n");
} else {
  train(model);
}

if (startChar && [...startChar].some(c => !(c in stoi))) {
  console.log(`输入含词表外的字（模型只认识语料中出现过的字）。`);
  console.log(`可用的字例如: 月 春 山 风 花 雪 江 云 日 空 松 深 明 ...`);
  process.exit(1);
}

console.log(startChar ? `以「${startChar}」开头生成 ${count} 首:\n` : `自由生成 ${count} 首:\n`);
for (let i = 0; i < count; i++) {
  // 上下文以 \n 开头（= "一首诗的开始"），生成到 \n 或 24 字为止
  const ctx = encode("\n" + startChar);
  const out = model.generate(ctx, 65 - startChar.length,   // 最长七律 64 字+句号，碰到 \n 自然停
    { temperature: 0.6, topK: 5, stopId: NL, explore: startChar.length < 2 });
  console.log(`  ${i + 1}. ${decode(out).trim()}`);
}

} // end require.main
