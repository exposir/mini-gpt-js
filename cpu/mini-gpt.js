// ============================================================
// mini-gpt.js —— 纯 JavaScript 实现的 GPT 风格 Transformer（前向传播 + 文本生成）
// 零依赖，node mini-gpt.js 直接运行
// 结构与 GPT-2 / nanoGPT 一致：
//   Token Embedding + Position Embedding
//   → N × [ LayerNorm → 多头因果自注意力 → 残差
//           LayerNorm → MLP(GELU)        → 残差 ]
//   → LayerNorm → LM Head → 下一个 token 的概率分布
// ============================================================

// ---------- 1. 基础数学工具（手写矩阵运算） ----------

// 矩阵用 "行数组 of Float32Array" 表示：A[i][j]
function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Float32Array(cols));
}

// 高斯随机初始化（Box-Muller），std 控制初始化幅度
function randn(rows, cols, std = 0.02) {
  const m = zeros(rows, cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const u = 1 - Math.random(), v = Math.random();
      m[i][j] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
    }
  return m;
}

// 矩阵乘法 C = A(n×k) · B(k×m) —— Transformer 90% 的计算量都在这
function matmul(A, B) {
  const n = A.length, k = B.length, m = B[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i][j] += a * B[p][j];
    }
  }
  return C;
}

// 逐行 softmax：把任意实数向量变成概率分布
function softmaxRow(row) {
  const max = Math.max(...row);              // 减最大值防止指数溢出
  const exps = row.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// GELU 激活函数（GPT 系列标配，比 ReLU 更平滑）
function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
}

// LayerNorm：对每个 token 的特征向量做归一化，训练稳定性的关键
function layerNorm(X, gain, bias) {
  const out = zeros(X.length, X[0].length);
  for (let i = 0; i < X.length; i++) {
    const row = X[i], d = row.length;
    let mean = 0;
    for (const v of row) mean += v / d;
    let variance = 0;
    for (const v of row) variance += (v - mean) ** 2 / d;
    const inv = 1 / Math.sqrt(variance + 1e-5);
    for (let j = 0; j < d; j++) out[i][j] = (row[j] - mean) * inv * gain[j] + bias[j];
  }
  return out;
}

// ---------- 2. 多头因果自注意力（Transformer 的灵魂） ----------

class CausalSelfAttention {
  constructor(nEmbd, nHead) {
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    // 四个投影矩阵：Q(查询)、K(键)、V(值)、输出投影
    this.Wq = randn(nEmbd, nEmbd);
    this.Wk = randn(nEmbd, nEmbd);
    this.Wv = randn(nEmbd, nEmbd);
    this.Wo = randn(nEmbd, nEmbd);
  }

  // X: (T × nEmbd)，T 是当前序列长度
  forward(X) {
    const T = X.length, { nHead, headDim } = this;
    const Q = matmul(X, this.Wq);
    const K = matmul(X, this.Wk);
    const V = matmul(X, this.Wv);
    const out = zeros(T, nHead * headDim);

    // 每个"头"独立地做一遍注意力，关注不同的语义子空间
    for (let h = 0; h < nHead; h++) {
      const off = h * headDim;
      for (let i = 0; i < T; i++) {
        // ① 打分：第 i 个 token 的 Q 与所有 j≤i 的 K 做点积
        //    因果掩码 = 只看过去（j ≤ i），不能偷看未来
        const scores = new Array(i + 1);
        for (let j = 0; j <= i; j++) {
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += Q[i][off + d] * K[j][off + d];
          scores[j] = dot / Math.sqrt(headDim);   // 缩放，防止梯度消失
        }
        // ② softmax 得到注意力权重："我该关注前文中的谁？关注多少？"
        const weights = softmaxRow(scores);
        // ③ 用权重对 V 加权求和 —— 信息按相关性汇聚过来
        for (let j = 0; j <= i; j++)
          for (let d = 0; d < headDim; d++)
            out[i][off + d] += weights[j] * V[j][off + d];
      }
    }
    return matmul(out, this.Wo);  // 拼接所有头后再投影一次
  }
}

// ---------- 3. MLP（前馈网络）：每个 token 独立做非线性变换 ----------

class MLP {
  constructor(nEmbd) {
    this.W1 = randn(nEmbd, 4 * nEmbd);  // 先升维到 4 倍（GPT 惯例）
    this.W2 = randn(4 * nEmbd, nEmbd);  // 再降回来
  }
  forward(X) {
    const H = matmul(X, this.W1);
    for (const row of H) for (let j = 0; j < row.length; j++) row[j] = gelu(row[j]);
    return matmul(H, this.W2);
  }
}

// ---------- 4. Transformer Block：注意力 + MLP，各带残差连接 ----------

class Block {
  constructor(nEmbd, nHead) {
    this.ln1g = new Float32Array(nEmbd).fill(1); this.ln1b = new Float32Array(nEmbd);
    this.ln2g = new Float32Array(nEmbd).fill(1); this.ln2b = new Float32Array(nEmbd);
    this.attn = new CausalSelfAttention(nEmbd, nHead);
    this.mlp = new MLP(nEmbd);
  }
  forward(X) {
    // x = x + attn(ln(x))   残差连接：信息高速公路，深层网络能训得动全靠它
    const a = this.attn.forward(layerNorm(X, this.ln1g, this.ln1b));
    for (let i = 0; i < X.length; i++)
      for (let j = 0; j < X[0].length; j++) X[i][j] += a[i][j];
    // x = x + mlp(ln(x))
    const m = this.mlp.forward(layerNorm(X, this.ln2g, this.ln2b));
    for (let i = 0; i < X.length; i++)
      for (let j = 0; j < X[0].length; j++) X[i][j] += m[i][j];
    return X;
  }
}

// ---------- 5. 完整 GPT 模型 ----------

class MiniGPT {
  constructor({ vocabSize, blockSize, nLayer, nHead, nEmbd }) {
    this.blockSize = blockSize;                    // 上下文窗口长度
    this.wte = randn(vocabSize, nEmbd);            // token 嵌入表：token id → 向量
    this.wpe = randn(blockSize, nEmbd);            // 位置嵌入表：位置 → 向量
    this.blocks = Array.from({ length: nLayer }, () => new Block(nEmbd, nHead));
    this.lnFg = new Float32Array(nEmbd).fill(1);
    this.lnFb = new Float32Array(nEmbd);
    // LM Head 与 wte 权重共享（GPT-2 同款技巧）：向量 → 词表上的打分
  }

  // 前向传播：token id 序列 → 最后一个位置的下一词 logits
  forward(ids) {
    const T = ids.length, nEmbd = this.wte[0].length;
    // token 嵌入 + 位置嵌入（模型知道"是什么词"和"在第几个位置"）
    let X = zeros(T, nEmbd);
    for (let i = 0; i < T; i++)
      for (let j = 0; j < nEmbd; j++)
        X[i][j] = this.wte[ids[i]][j] + this.wpe[i][j];

    for (const block of this.blocks) X = block.forward(X);   // 层层堆叠
    X = layerNorm(X, this.lnFg, this.lnFb);

    // 只取最后一个 token 的向量，与嵌入表做点积 → 每个候选词的得分
    const last = X[T - 1];
    return this.wte.map(row => {
      let dot = 0;
      for (let j = 0; j < nEmbd; j++) dot += last[j] * row[j];
      return dot;
    });
  }

  // 自回归生成：预测一个 → 拼回去 → 再预测下一个
  generate(ids, maxNewTokens, temperature = 1.0) {
    ids = [...ids];
    for (let step = 0; step < maxNewTokens; step++) {
      const ctx = ids.slice(-this.blockSize);          // 超长就截断到窗口内
      const logits = this.forward(ctx);
      const probs = softmaxRow(logits.map(l => l / temperature));
      // 按概率分布随机采样（这就是"同一问题每次回答不同"的原因）
      let r = Math.random(), next = 0;
      for (let i = 0; i < probs.length; i++) { r -= probs[i]; if (r <= 0) { next = i; break; } }
      ids.push(next);
    }
    return ids;
  }

  paramCount() {
    let n = this.wte.length * this.wte[0].length + this.wpe.length * this.wpe[0].length;
    for (const b of this.blocks) {
      for (const W of [b.attn.Wq, b.attn.Wk, b.attn.Wv, b.attn.Wo, b.mlp.W1, b.mlp.W2])
        n += W.length * W[0].length;
      n += b.ln1g.length * 2 + b.ln2g.length * 2;
    }
    return n + this.lnFg.length * 2;
  }
}

// ---------- 6. 跑起来：字符级词表 + 随机权重生成 ----------

const text = "hello transformer! attention is all you need.";
const chars = [...new Set(text)].sort();                       // 去重字符 = 词表
const stoi = Object.fromEntries(chars.map((c, i) => [c, i]));  // 字符 → id
const itos = Object.fromEntries(chars.map((c, i) => [i, c]));  // id → 字符
const encode = s => [...s].map(c => stoi[c]);
const decode = ids => ids.map(i => itos[i]).join("");

const model = new MiniGPT({
  vocabSize: chars.length,  // 词表大小（这里只有 ~20 个字符）
  blockSize: 32,            // 上下文窗口
  nLayer: 2,                // 层数（GPT-3 是 96 层）
  nHead: 4,                 // 注意力头数
  nEmbd: 64,                // 嵌入维度（GPT-3 是 12288）
});

console.log(`词表: ${JSON.stringify(chars.join(""))}  (${chars.length} 个字符)`);
console.log(`参数量: ${model.paramCount().toLocaleString()}（GPT-3 是 1750 亿，原理完全相同）\n`);

const prompt = "hello ";
const out = model.generate(encode(prompt), 40, 0.8);
console.log(`输入 prompt : "${prompt}"`);
console.log(`模型续写   : "${decode(out)}"`);
console.log("\n※ 权重是随机的、没训练过，所以输出是乱码 ——");
console.log("  训练的全部意义，就是把这些随机矩阵调成能算出正确概率的数值。");
