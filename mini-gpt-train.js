// ============================================================
// mini-gpt-train.js —— 可训练的纯 JS GPT：前向 + 反向传播 + Adam 优化器
// 零依赖，node mini-gpt-train.js 直接运行
//
// 训练原理（和 GPT-3 完全一样，只是规模小 100 万倍）：
//   ① 前向传播：对文本每个位置预测"下一个字符"的概率分布
//   ② 交叉熵损失：正确字符的概率越低，损失越大
//   ③ 反向传播：链式法则算出每个参数对损失的梯度（该往哪边调）
//   ④ Adam 更新：参数沿梯度反方向走一小步
//   ⑤ 重复几千次 → 随机矩阵逐渐变成"会预测"的矩阵
// ============================================================

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
function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(GC * (x + GA * x ** 3)));
}
// GELU 的导数（反向传播要用）
function geluGrad(x) {
  const t = Math.tanh(GC * (x + GA * x ** 3));
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GC * (1 + 3 * GA * x * x);
}

// ---------- 2. 微型 autograd 引擎（PyTorch 的核心思想，浓缩成 30 行） ----------
// 每个 Tensor 记住：数值 data、梯度 grad、它由谁算出来（_parents）、
// 以及"怎么把自己的梯度传回给爹妈"（_backward 闭包）。

class Tensor {
  constructor(data) {
    this.data = data;                                   // 数值（前向传播算出）
    this.grad = data.map(r => new Float32Array(r.length)); // 梯度（反向传播算出）
    this._backward = () => {};
    this._parents = [];
  }
}

// 从 loss 出发，按拓扑序反向执行每个节点的 _backward —— 这就是"反向传播"
function backward(loss) {
  const topo = [], visited = new Set();
  (function build(t) {
    if (visited.has(t)) return;
    visited.add(t);
    for (const p of t._parents) build(p);
    topo.push(t);
  })(loss);
  loss.grad[0][0] = 1;                                  // dL/dL = 1，链式法则的起点
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
}

// ---------- 3. 带反向传播的算子 ----------
// 每个算子 = 前向计算 + 一个"梯度怎么回传"的闭包（就是手推的求导公式）

// C = A·B      反向：dA = dC·Bᵀ, dB = Aᵀ·dC
function matmul(A, B) {
  const n = A.data.length, k = B.data.length, m = B.data[0].length;
  const C = new Tensor(zeros(n, m));
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      const a = A.data[i][p];
      for (let j = 0; j < m; j++) C.data[i][j] += a * B.data[p][j];
    }
  C._parents = [A, B];
  C._backward = () => {
    for (let i = 0; i < n; i++)
      for (let p = 0; p < k; p++) {
        let s = 0;
        for (let j = 0; j < m; j++) {
          s += C.grad[i][j] * B.data[p][j];
          B.grad[p][j] += A.data[i][p] * C.grad[i][j];
        }
        A.grad[i][p] += s;
      }
  };
  return C;
}

// C = A·Wᵀ （LM Head 用，权重与嵌入表共享）
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

// C = A + B（残差连接用）    反向：梯度原样分给两边
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

// 逐元素 GELU    反向：乘上 gelu'(x)
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

// LayerNorm（g、b 是 1×d 的可学习参数）
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

// 多头因果自注意力（融合算子，反向传播是手推的 softmax-attention 求导公式）
function causalAttention(Q, K, V, nHead) {
  const T = Q.data.length, nEmbd = Q.data[0].length, hd = nEmbd / nHead;
  const C = new Tensor(zeros(T, nEmbd));
  const attnW = [];                       // 保存注意力权重，反向传播要用
  for (let h = 0; h < nHead; h++) {
    const off = h * hd; attnW[h] = [];
    for (let i = 0; i < T; i++) {
      const scores = new Array(i + 1);    // 因果掩码：只算 j ≤ i
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
            dA[j] += C.grad[i][off + d] * V.data[j][off + d];   // dL/d注意力权重
            V.grad[j][off + d] += w[j] * C.grad[i][off + d];    // dL/dV
          }
        let sum = 0;                       // softmax 反向：dS = w⊙(dA - Σw·dA)
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

// 嵌入查表：X[i] = wte[ids[i]] + wpe[i]    反向：梯度散射回对应行
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

// 交叉熵损失：-log(正确字符的预测概率)，对所有位置取平均
// 反向传播出奇地优雅：dlogits = 预测概率 - 正确答案的 one-hot
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

// ---------- 4. GPT 模型（结构与上一版完全一致，只是换成 autograd 算子） ----------

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
      X = add(X, matmul(attnOut, b.Wo));                 // 残差 ①
      const h2 = layerNorm(X, b.ln2g, b.ln2b);
      X = add(X, matmul(geluOp(matmul(h2, b.W1)), b.W2)); // 残差 ②
    }
    X = layerNorm(X, this.lnFg, this.lnFb);
    return matmulT(X, this.wte);                          // LM Head（权重共享）
  }

  // 贪心生成：每步选概率最高的字符（训练好后应精确复现训练文本）
  generate(ids, maxNewTokens) {
    ids = [...ids];
    for (let s = 0; s < maxNewTokens; s++) {
      const logits = this.forward(ids.slice(-this.blockSize));
      const last = logits.data[logits.data.length - 1];
      let best = 0;
      for (let j = 1; j < last.length; j++) if (last[j] > last[best]) best = j;
      ids.push(best);
    }
    return ids;
  }
}

// ---------- 5. Adam 优化器（大模型训练的标配） ----------

class Adam {
  constructor(params, lr = 3e-3) {
    this.params = params; this.lr = lr; this.t = 0;
    this.m = params.map(p => p.data.map(r => new Float32Array(r.length)));
    this.v = params.map(p => p.data.map(r => new Float32Array(r.length)));
  }
  zeroGrad() {
    for (const p of this.params) for (const row of p.grad) row.fill(0);
  }
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

// ---------- 6. 训练！ ----------

const text = "hello transformer! attention is all you need.";
const chars = [...new Set(text)].sort();
const stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
const itos = Object.fromEntries(chars.map((c, i) => [i, c]));
const encode = s => [...s].map(c => stoi[c]);
const decode = ids => ids.map(i => itos[i]).join("");

const model = new MiniGPT({
  vocabSize: chars.length, blockSize: 64, nLayer: 2, nHead: 4, nEmbd: 48,
});
const opt = new Adam(model.params(), 3e-3);

// 训练数据：输入是文本[0..n-1]，目标是文本[1..n]（每个位置预测下一个字符）
const ids = encode(text);
const xs = ids.slice(0, -1), ys = ids.slice(1);

const prompt = "hello ";
console.log(`训练文本 : "${text}"`);
console.log(`训练前生成: "${decode(model.generate(encode(prompt), 39))}"\n`);

const t0 = Date.now();
for (let step = 1; step <= 1200; step++) {
  opt.zeroGrad();                        // 清空上一步的梯度
  const loss = crossEntropy(model.forward(xs), ys);  // ① 前向 + ② 算损失
  backward(loss);                        // ③ 反向传播
  opt.step();                            // ④ 更新参数
  if (step % 150 === 0 || step === 1)
    console.log(`step ${String(step).padStart(4)}  loss = ${loss.data[0][0].toFixed(4)}`);
}
console.log(`\n训练耗时: ${((Date.now() - t0) / 1000).toFixed(1)}s`);

console.log(`训练后生成: "${decode(model.generate(encode(prompt), 39))}"`);
console.log("\n※ loss 从 ~3（等于瞎猜 ln(18)≈2.9）降到接近 0，");
console.log("  同一套随机矩阵被梯度下降『捏』成了会背这句话的模型。");
console.log("  GPT 的训练与此完全同构：文本换成整个互联网，参数放大 100 万倍。");
