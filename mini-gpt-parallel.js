// ============================================================
// mini-gpt-parallel.js —— Worker 数据并行训练器（零依赖，Node 内置能力）
//
// 原理：串行版的"梯度累积 batch=8"天然是并行结构 ——
//   串行: 1 线程依次算 8 首诗梯度 → 求和 → Adam 更新
//   并行: 8 个 Worker 同时各算 1 首 → 主线程求和 → Adam 更新
// 数学上完全等价，速度 ≈ 核数倍。
//
// 共享内存布局：
//   weightsSAB          全部参数展平为一条 Float32Array，8 个 Worker 零拷贝共享（只读）
//   gradSAB × 8         每个 Worker 独占一条梯度缓冲（写），主线程合并（读）
//
// 用法: node mini-gpt-parallel.js [目标总步数]   （默认续到权重里记录的 totalSteps）
// ============================================================

const { Worker, isMainThread, parentPort, workerData } = require("worker_threads");
const fs = require("fs");
const {
  MiniGPT, CFG, encode, loadWeights, saveWeights, WEIGHTS, backward, crossEntropy,
} = require("./mini-gpt-poet.js");
const POEMS = require("./poems.js");

const N_WORKERS = 8;              // 每轮 8 首诗 = 与串行版 batch=8 完全等价
const BASE_LR = 1e-3;             // 与串行续训版一致

// ---------- 参数展平：把模型 44 个矩阵映射到一条扁平数组上 ----------

function layout(model) {
  const spans = [];
  let offset = 0;
  for (const p of model.params()) {
    const rows = p.data.length, cols = p.data[0].length;
    spans.push({ offset, rows, cols });
    offset += rows * cols;
  }
  return { spans, total: offset };
}

// 把模型参数的 data（或 grad）替换为扁平数组上的零拷贝视图
function attach(model, flat, field) {
  const { spans } = layout(model);
  model.params().forEach((p, k) => {
    const { offset, rows, cols } = spans[k];
    const view = [];
    for (let i = 0; i < rows; i++)
      view.push(flat.subarray(offset + i * cols, offset + (i + 1) * cols));
    p[field] = view;
  });
}

// 一首诗的前向+反向（Worker 与主线程自检共用）
function lossOnPoem(model, poem, withBackward) {
  const ids = encode("\n" + poem + "\n");
  const loss = crossEntropy(model.forward(ids.slice(0, -1)), ids.slice(1));
  if (withBackward) backward(loss);
  return loss.data[0][0];
}

// ============================================================
// Worker 分支：收到诗的下标 → 清梯度 → 算前向反向 → 报告 loss
// ============================================================

if (!isMainThread) {
  const model = new MiniGPT(CFG);
  const wView = new Float32Array(workerData.weightsSAB);
  const gView = new Float32Array(workerData.gradSAB);
  attach(model, wView, "data");   // 权重：共享只读视图
  attach(model, gView, "grad");   // 梯度：本 Worker 独占缓冲

  parentPort.on("message", ({ poemIdxs }) => {
    gView.fill(0);                // 清上一轮梯度
    let sum = 0;
    for (const idx of poemIdxs) sum += lossOnPoem(model, POEMS[idx], true);
    parentPort.postMessage({ loss: sum / poemIdxs.length });
  });
  return;
}

// ============================================================
// 主线程：装载权重 → 起 Worker → 调度循环 → Adam 更新 → checkpoint
// ============================================================

// 每个 Worker 一轮算 POEMS_PER_WORKER 首诗（加大任务粒度，摄薄通信开销）
const POEMS_PER_WORKER = 4;   // 每轮 batch = N_WORKERS × 4 = 32

// 扁平化 Adam（主线程在共享权重上原地更新）
class FlatAdam {
  constructor(n, lr) {
    this.lr = lr; this.t = 0;
    this.m = new Float32Array(n);
    this.v = new Float32Array(n);
  }
  step(w, gradViews, lr) {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    const K = gradViews.length;
    for (let i = 0; i < w.length; i++) {
      let g = 0;
      for (let k = 0; k < K; k++) g += gradViews[k][i];   // 合并全部 Worker 梯度
      this.m[i] = b1 * this.m[i] + (1 - b1) * g;
      this.v[i] = b2 * this.v[i] + (1 - b2) * g * g;
      w[i] -= lr * (this.m[i] / c1) / (Math.sqrt(this.v[i] / c2) + eps);
    }
  }
}

(async () => {
  // 1. 装载权重（有则续训，无则从零开始）
  const model = new MiniGPT(CFG);
  let startStep = 0;
  let totalSteps = Number(process.argv[2]) || 300000;
  if (fs.existsSync(WEIGHTS)) {
    const meta = loadWeights(model);
    startStep = meta.step || 0;
    totalSteps = Number(process.argv[2]) || meta.totalSteps || startStep + 200000;
  } else {
    console.log("无权重文件，从随机初始化从零训练");
  }
  if (startStep >= totalSteps) { console.log(`已到 ${startStep} 步，无需训练`); process.exit(0); }

  // 2. 权重拷入共享内存，主线程模型切到共享视图
  const { total } = layout(model);
  const weightsSAB = new SharedArrayBuffer(total * 4);
  const w = new Float32Array(weightsSAB);
  let cursor = 0;
  for (const p of model.params()) for (const row of p.data) { w.set(row, cursor); cursor += row.length; }
  attach(model, w, "data");

  console.log(`并行训练器: ${N_WORKERS} Workers × ${POEMS_PER_WORKER} 首/轮 | 参数 ${total.toLocaleString()}`);
  console.log(`续训 ${startStep} → ${totalSteps} 步 (lr=${BASE_LR}, 每轮 batch=${N_WORKERS * POEMS_PER_WORKER})`);

  // 3. 启动 Workers
  const gradSABs = [], gradViews = [], workers = [];
  for (let k = 0; k < N_WORKERS; k++) {
    const gradSAB = new SharedArrayBuffer(total * 4);
    gradSABs.push(gradSAB);
    gradViews.push(new Float32Array(gradSAB));
    workers.push(new Worker(__filename, { workerData: { weightsSAB, gradSAB } }));
  }
  const ask = (worker, poemIdxs) => new Promise(res => {
    worker.once("message", res);
    worker.postMessage({ poemIdxs });
  });

  // 4. 正确性自检：主线程与 Worker0 对同一首诗算 loss，必须一致
  const probeLoss = lossOnPoem(model, POEMS[0], false);
  const { loss: workerLoss } = await ask(workers[0], [0]);
  const diff = Math.abs(probeLoss - workerLoss);
  console.log(`自检: 主线程 loss=${probeLoss.toFixed(6)} vs Worker loss=${workerLoss.toFixed(6)} 差=${diff.toExponential(2)}`);
  if (diff > 1e-3) { console.error("自检失败，中止！"); process.exit(1); }
  console.log("自检通过，开始训练\n");

  // 5. 训练主循环（单阶段：8 Worker 并行各算 4 首 → 主线程合并 Adam）
  const adam = new FlatAdam(total, BASE_LR);
  const t0 = Date.now();
  let step = startStep, lossSum = 0, lossN = 0;
  let nextLog = Math.ceil((step + 1) / 5000) * 5000;
  let nextCkpt = Math.ceil((step + 1) / 10000) * 10000;

  while (step < totalSteps) {
    const lr = step > totalSteps * 0.6 ? BASE_LR / 3 : BASE_LR;
    // 8 Worker 并行，每个算 POEMS_PER_WORKER 首（梯度在 Worker 内部累加）
    const jobs = workers.map(wk => {
      const idxs = [];
      for (let i = 0; i < POEMS_PER_WORKER; i++) idxs.push(Math.floor(Math.random() * POEMS.length));
      return ask(wk, idxs);
    });
    const results = await Promise.all(jobs);
    adam.step(w, gradViews, lr);

    step += N_WORKERS * POEMS_PER_WORKER;
    lossSum += results.reduce((s, r) => s + r.loss, 0) / N_WORKERS; lossN++;

    if (step >= nextLog) {
      const speed = (Date.now() - t0) / 1000 / (step - startStep);
      console.log(`  step ${step}  平均loss = ${(lossSum / lossN).toFixed(4)}  (${speed.toFixed(4)}s/样本)`);
      lossSum = 0; lossN = 0; nextLog += 5000;
    }
    if (step >= nextCkpt) {
      saveWeights(model, step, totalSteps);
      console.log(`  [checkpoint] 已存盘 @ step ${step}`);
      nextCkpt += 10000;
    }
  }

  saveWeights(model, totalSteps, totalSteps);
  const hrs = (Date.now() - t0) / 3600000;
  console.log(`\n训练完成，耗时 ${hrs.toFixed(2)} 小时，权重已存盘`);
  for (const wk of workers) wk.terminate();
})();
