// ============================================================
// gpu-train.js —— 终极形态·阶段2：全 GPU 训练器（Deno + WebGPU）
//
// 设计：
//   - 全部算子 WGSL 化：embed / layerNorm / attention / GELU /
//     matmul(NN/NT/TN) / crossEntropy / Adam
//   - 权重·梯度·Adam 状态全程常驻显存，每步 CPU 只上传 batch 的 token id
//   - batch = 32 首诗（800 token）一次前向/反向
//   - 权重 JSON 与 mini-gpt-poet.js 完全互通（可互相续训/推理）
//
// 用法: deno run --allow-read --allow-write gpu-train.js [目标样本数]
// ============================================================

import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
// 语料、词表、训练/验证切分统一从这里来（切分逻辑单一来源，防止各处抄错）
// 注意取的是 *_IDX（下标数组），uploadBatch 靠下标去 POEMS 里取诗
import { POEMS, chars, stoi, encode, TRAIN_IDX as TRAIN, VAL_IDX as VAL, CORPUS_NAME } from "./data-split.js";

// ---------- 配置 ----------
const CFG = { vocabSize: chars.length, blockSize: 66, nLayer: 10, nHead: 10, nEmbd: 640 };
const E = CFG.nEmbd, H = CFG.nHead, HD = E / H, V = CFG.vocabSize, FF = 4 * E;
const T = CFG.blockSize - 1;     // 最长诗（七律）66 token → 输入 65；短诗 padding+掩码
const B = 32, BT = B * T;        // batch
const PAD = 0xffffffff;          // 目标位哨兵值：超出词表即视为 padding，不计损失
const BASE_LR = 5e-4;            // 640 维大模型配小学习率（1e-3 在 10 万步实测平台震荡）
// 权重文件前缀可由第 2 个参数指定，避免覆盖线上正式权重
const PREFIX = Deno.args[1] || "poet-weights";
const WEIGHTS = new URL(`./${PREFIX}.json`, import.meta.url).pathname;   // 仅兼容旧档
const META = new URL(`./${PREFIX}.meta.json`, import.meta.url).pathname;
const BIN = new URL(`./${PREFIX}.bin`, import.meta.url).pathname;
const CURVE = new URL(`./${PREFIX}-curve.csv`, import.meta.url).pathname;
const EVAL_EVERY = Number(Deno.args[2]) || 25000;   // 每 2.5 万步在验证集上评一次

// ============================================================
// WGSL 内核
// ============================================================

// --- 矩阵乘（V2 寄存器分块，NN/NT/TN 三变体，带累加开关） ---
const MM_WGSL = /* wgsl */ `
struct Dims { n: u32, k: u32, m: u32, accum: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> tA: array<f32, 1024>;
var<workgroup> tB: array<f32, 1024>;

fn body(wid: vec3<u32>, lid: vec3<u32>, mode: u32) {
  let rowBase = wid.y * 64u + lid.y * 4u;
  let colBase = wid.x * 64u + lid.x * 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  let numTiles = (dims.k + 15u) / 16u;
  let tid = lid.y * 16u + lid.x;

  for (var t = 0u; t < numTiles; t++) {
    for (var s = 0u; s < 4u; s++) {
      let idx = tid * 4u + s;
      let ar = idx / 16u; let ac = idx % 16u;      // A 块: 64行(n) × 16列(k)
      let aRow = wid.y * 64u + ar; let aCol = t * 16u + ac;
      var av = 0.0;
      if (aRow < dims.n && aCol < dims.k) {
        if (mode == 2u) { av = A[aCol * dims.n + aRow]; }   // TN: A 是 (k×n)
        else            { av = A[aRow * dims.k + aCol]; }   // NN/NT: A 是 (n×k)
      }
      tA[idx] = av;
      let br = idx / 64u; let bc = idx % 64u;      // B 块: 16行(k) × 64列(m)
      let bRow = t * 16u + br; let bCol = wid.x * 64u + bc;
      var bv = 0.0;
      if (bRow < dims.k && bCol < dims.m) {
        if (mode == 1u) { bv = B[bCol * dims.k + bRow]; }   // NT: B 是 (m×k)
        else            { bv = B[bRow * dims.m + bCol]; }   // NN/TN: B 是 (k×m)
      }
      tB[idx] = bv;
    }
    workgroupBarrier();
    for (var i = 0u; i < 16u; i++) {
      let a0 = tA[(lid.y*4u+0u)*16u+i]; let a1 = tA[(lid.y*4u+1u)*16u+i];
      let a2 = tA[(lid.y*4u+2u)*16u+i]; let a3 = tA[(lid.y*4u+3u)*16u+i];
      let b0 = tB[i*64u+lid.x*4u+0u];   let b1 = tB[i*64u+lid.x*4u+1u];
      let b2 = tB[i*64u+lid.x*4u+2u];   let b3 = tB[i*64u+lid.x*4u+3u];
      acc[0]+=a0*b0; acc[1]+=a0*b1; acc[2]+=a0*b2; acc[3]+=a0*b3;
      acc[4]+=a1*b0; acc[5]+=a1*b1; acc[6]+=a1*b2; acc[7]+=a1*b3;
      acc[8]+=a2*b0; acc[9]+=a2*b1; acc[10]+=a2*b2; acc[11]+=a2*b3;
      acc[12]+=a3*b0; acc[13]+=a3*b1; acc[14]+=a3*b2; acc[15]+=a3*b3;
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < 4u; r++) {
    let row = rowBase + r;
    if (row >= dims.n) { continue; }
    for (var c = 0u; c < 4u; c++) {
      let col = colBase + c;
      if (col < dims.m) {
        let idx = row * dims.m + col;
        let prev = select(0.0, C[idx], dims.accum == 1u);
        C[idx] = prev + acc[r * 4u + c];
      }
    }
  }
}
@compute @workgroup_size(16,16) fn mm_nn(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) { body(w, l, 0u); }
@compute @workgroup_size(16,16) fn mm_nt(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) { body(w, l, 1u); }
@compute @workgroup_size(16,16) fn mm_tn(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) { body(w, l, 2u); }
`;

// --- 因果注意力（前向 + 三段反向，无原子操作设计） ---
const ATTN_WGSL = /* wgsl */ `
struct AD { nb: u32, nh: u32, t: u32, e: u32 };
@group(0) @binding(0) var<uniform> d: AD;
@group(0) @binding(1) var<storage, read> P1: array<f32>;
@group(0) @binding(2) var<storage, read> P2: array<f32>;
@group(0) @binding(3) var<storage, read> P3: array<f32>;
@group(0) @binding(4) var<storage, read_write> O1: array<f32>;
@group(0) @binding(5) var<storage, read_write> O2: array<f32>;

// fwd: P1=Q P2=K P3=V, O1=attnW, O2=out       线程=(b,h,i)
@compute @workgroup_size(64) fn attn_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let hd = d.e / d.nh;
  let total = d.nb * d.nh * d.t;
  if (g.x >= total) { return; }
  let i = g.x % d.t; let h = (g.x / d.t) % d.nh; let b = g.x / (d.t * d.nh);
  let off = h * hd;
  let scale = 1.0 / sqrt(f32(hd));
  var scores: array<f32, 72>;
  var mx = -1e30;
  for (var j = 0u; j <= i; j++) {
    var dot = 0.0;
    for (var k = 0u; k < hd; k++) {
      dot += P1[(b*d.t+i)*d.e+off+k] * P2[(b*d.t+j)*d.e+off+k];
    }
    scores[j] = dot * scale;
    mx = max(mx, scores[j]);
  }
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) { scores[j] = exp(scores[j] - mx); sum += scores[j]; }
  let wBase = ((b*d.nh+h)*d.t+i)*d.t;
  for (var j = 0u; j <= i; j++) { O1[wBase+j] = scores[j] / sum; }
  for (var k = 0u; k < hd; k++) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc += O1[wBase+j] * P3[(b*d.t+j)*d.e+off+k]; }
    O2[(b*d.t+i)*d.e+off+k] = acc;
  }
}

// dv: P1=attnW P2=dOut, O1=dV                 线程=(b,h,j)
@compute @workgroup_size(64) fn attn_dv(@builtin(global_invocation_id) g: vec3<u32>) {
  let hd = d.e / d.nh;
  let total = d.nb * d.nh * d.t;
  if (g.x >= total) { return; }
  let j = g.x % d.t; let h = (g.x / d.t) % d.nh; let b = g.x / (d.t * d.nh);
  let off = h * hd;
  for (var k = 0u; k < hd; k++) {
    var acc = 0.0;
    for (var i = j; i < d.t; i++) {
      acc += P1[((b*d.nh+h)*d.t+i)*d.t+j] * P2[(b*d.t+i)*d.e+off+k];
    }
    O1[(b*d.t+j)*d.e+off+k] = acc;
  }
}

// dsq: P1=attnW P2=dOut P3=V(算dA) + K(经O2侧读不行) —— 拆两缓冲:
//   这里 P3=V, O1=dS(输出), O2=dQ(输出);  K 复用 P1? 不行 —— 改为两趟:
//   本内核输出 dS; dQ 用 mm 无法表达(因果+分头), 直接在这里读 K:
//   绑定复用: P1=attnW P2=dOut P3=V | O1=dS O2=dQ, K 通过 binding6
@group(0) @binding(6) var<storage, read> P4: array<f32>;
@compute @workgroup_size(64) fn attn_dsq(@builtin(global_invocation_id) g: vec3<u32>) {
  let hd = d.e / d.nh;
  let total = d.nb * d.nh * d.t;
  if (g.x >= total) { return; }
  let i = g.x % d.t; let h = (g.x / d.t) % d.nh; let b = g.x / (d.t * d.nh);
  let off = h * hd;
  let wBase = ((b*d.nh+h)*d.t+i)*d.t;
  var dA: array<f32, 72>;
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) {
    var acc = 0.0;
    for (var k = 0u; k < hd; k++) {
      acc += P2[(b*d.t+i)*d.e+off+k] * P3[(b*d.t+j)*d.e+off+k];
    }
    dA[j] = acc;
    sum += P1[wBase+j] * acc;
  }
  let scale = 1.0 / sqrt(f32(hd));
  for (var j = 0u; j <= i; j++) {
    O1[wBase+j] = P1[wBase+j] * (dA[j] - sum) * scale;
  }
  for (var k = 0u; k < hd; k++) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc += O1[wBase+j] * P4[(b*d.t+j)*d.e+off+k]; }
    O2[(b*d.t+i)*d.e+off+k] = acc;
  }
}

// dk: P1=dS P2=Q, O1=dK                       线程=(b,h,j)
@compute @workgroup_size(64) fn attn_dk(@builtin(global_invocation_id) g: vec3<u32>) {
  let hd = d.e / d.nh;
  let total = d.nb * d.nh * d.t;
  if (g.x >= total) { return; }
  let j = g.x % d.t; let h = (g.x / d.t) % d.nh; let b = g.x / (d.t * d.nh);
  let off = h * hd;
  for (var k = 0u; k < hd; k++) {
    var acc = 0.0;
    for (var i = j; i < d.t; i++) {
      acc += P1[((b*d.nh+h)*d.t+i)*d.t+j] * P2[(b*d.t+i)*d.e+off+k];
    }
    O1[(b*d.t+j)*d.e+off+k] = acc;
  }
}
`;

// --- LayerNorm ---
const LN_WGSL = /* wgsl */ `
struct LD { rows: u32, cols: u32, accum: u32, pad: u32 };
@group(0) @binding(0) var<uniform> d: LD;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read> Bb: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> XH: array<f32>;
@group(0) @binding(6) var<storage, read_write> IS: array<f32>;

@compute @workgroup_size(64) fn ln_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= d.rows) { return; }
  var mean = 0.0;
  for (var j = 0u; j < d.cols; j++) { mean += X[i*d.cols+j]; }
  mean /= f32(d.cols);
  var va = 0.0;
  for (var j = 0u; j < d.cols; j++) { let t = X[i*d.cols+j]-mean; va += t*t; }
  let inv = 1.0 / sqrt(va / f32(d.cols) + 1e-5);
  IS[i] = inv;
  for (var j = 0u; j < d.cols; j++) {
    let xh = (X[i*d.cols+j] - mean) * inv;
    XH[i*d.cols+j] = xh;
    Y[i*d.cols+j] = xh * G[j] + Bb[j];
  }
}
// dx: X=dY G=g XH=xhat IS=invstd → Y=dX(accum开关)
@compute @workgroup_size(64) fn ln_dx(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= d.rows) { return; }
  var m1 = 0.0; var m2 = 0.0;
  for (var j = 0u; j < d.cols; j++) {
    let dyh = X[i*d.cols+j] * G[j];
    m1 += dyh; m2 += dyh * XH[i*d.cols+j];
  }
  m1 /= f32(d.cols); m2 /= f32(d.cols);
  for (var j = 0u; j < d.cols; j++) {
    let dyh = X[i*d.cols+j] * G[j];
    let dx = IS[i] * (dyh - m1 - XH[i*d.cols+j] * m2);
    let idx = i*d.cols+j;
    let prev = select(0.0, Y[idx], d.accum == 1u);
    Y[idx] = prev + dx;
  }
}
// dgb: X=dY XH=xhat → Y=dg IS(借用)=db, 线程=列
@compute @workgroup_size(64) fn ln_dgb(@builtin(global_invocation_id) g: vec3<u32>) {
  let j = g.x;
  if (j >= d.cols) { return; }
  var dg = 0.0; var db = 0.0;
  for (var i = 0u; i < d.rows; i++) {
    dg += X[i*d.cols+j] * XH[i*d.cols+j];
    db += X[i*d.cols+j];
  }
  Y[j] += dg;
  IS[j] += db;
}
`;

// --- 杂项：嵌入 / GELU / add / 交叉熵 / Adam ---
const MISC_WGSL = /* wgsl */ `
struct MD { a: u32, b: u32, c: u32, pad: u32 };
@group(0) @binding(0) var<uniform> d: MD;
@group(0) @binding(1) var<storage, read> IU: array<u32>;
@group(0) @binding(2) var<storage, read> F1: array<f32>;
@group(0) @binding(3) var<storage, read> F2: array<f32>;
@group(0) @binding(4) var<storage, read_write> W1: array<f32>;
@group(0) @binding(5) var<storage, read_write> W2: array<f32>;

// embed_fwd: a=BT b=T c=E | IU=ids F1=wte F2=wpe W1=X0    线程=元素
@compute @workgroup_size(256) fn embed_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= d.a * d.c) { return; }
  let p = idx / d.c; let e = idx % d.c;
  W1[idx] = F1[IU[p]*d.c+e] + F2[(p % d.b)*d.c+e];
}
// embed_dwte: a=BT b=V c=E | IU=ids F1=dX0 W1=dwte(+=)     线程=(v,e)
@compute @workgroup_size(256) fn embed_dwte(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= d.b * d.c) { return; }
  let v = idx / d.c; let e = idx % d.c;
  var acc = 0.0;
  for (var p = 0u; p < d.a; p++) {
    if (IU[p] == v) { acc += F1[p*d.c+e]; }
  }
  W1[idx] += acc;
}
// embed_dwpe: a=B b=T c=E | F1=dX0 W1=dwpe(+=)             线程=(t,e)
@compute @workgroup_size(256) fn embed_dwpe(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= d.b * d.c) { return; }
  let t = idx / d.c; let e = idx % d.c;
  var acc = 0.0;
  for (var b = 0u; b < d.a; b++) { acc += F1[(b*d.b+t)*d.c+e]; }
  W1[idx] += acc;
}
// add: a=count | F1 F2 → W1                                 线程=元素
@compute @workgroup_size(256) fn addv(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= d.a) { return; }
  W1[g.x] = F1[g.x] + F2[g.x];
}
// gelu_fwd: a=count | F1=X → W1=Y（tanh 参数阔位：f32 下 e^²ˣ 溢出会产生 NaN）
@compute @workgroup_size(256) fn gelu_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= d.a) { return; }
  let x = F1[g.x];
  let GC = 0.7978845608028654; let GA = 0.044715;
  let t = clamp(GC * (x + GA * x*x*x), -9.0, 9.0);
  W1[g.x] = 0.5 * x * (1.0 + tanh(t));
}
// gelu_bwd: a=count | F1=X F2=dY → W1=dX
@compute @workgroup_size(256) fn gelu_bwd(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= d.a) { return; }
  let x = F1[g.x];
  let GC = 0.7978845608028654; let GA = 0.044715;
  let t = tanh(clamp(GC * (x + GA * x*x*x), -9.0, 9.0));
  let grad = 0.5*(1.0+t) + 0.5*x*(1.0-t*t)*GC*(1.0+3.0*GA*x*x);
  W1[g.x] = F2[g.x] * grad;
}
// ce: a=rows b=V c=有效位置数 | IU=targets F1=logits W1=dLogits W2=loss
//     目标 >= V 视为 padding：损失与梯度全部置零
@compute @workgroup_size(64) fn ce(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= d.a) { return; }
  let tgt = IU[i];
  if (tgt >= d.b) {                       // padding 行
    W2[i] = 0.0;
    for (var j = 0u; j < d.b; j++) { W1[i*d.b+j] = 0.0; }
    return;
  }
  var mx = -1e30;
  for (var j = 0u; j < d.b; j++) { mx = max(mx, F1[i*d.b+j]); }
  var sum = 0.0;
  for (var j = 0u; j < d.b; j++) { sum += exp(F1[i*d.b+j] - mx); }
  W2[i] = -(F1[i*d.b+tgt] - mx - log(sum));
  let scale = 1.0 / f32(d.c);
  for (var j = 0u; j < d.b; j++) {
    var p = exp(F1[i*d.b+j] - mx) / sum;
    if (j == tgt) { p -= 1.0; }
    W1[i*d.b+j] = p * scale;
  }
}
// adam: a=count | F1(借uniform经由MD无法传float) —— lr等经 F2[0..2] 传入
//   F2 = [lr, c1, c2] | W1=权重 W2=梯度, IU 无用; m/v 经 binding6/7
@group(0) @binding(6) var<storage, read_write> AM: array<f32>;
@group(0) @binding(7) var<storage, read_write> AV: array<f32>;
@compute @workgroup_size(256) fn adam(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= d.a) { return; }
  let gr = W2[g.x];
  let m = 0.9 * AM[g.x] + 0.1 * gr;
  let v = 0.999 * AV[g.x] + 0.001 * gr * gr;
  AM[g.x] = m; AV[g.x] = v;
  W1[g.x] -= F2[0] * (m / F2[1]) / (sqrt(v / F2[2]) + 1e-8);
}
`;

// ============================================================
// GPU 编排
// ============================================================

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  },
});

const pipelines = {};
for (const [src, names] of [
  [MM_WGSL, ["mm_nn", "mm_nt", "mm_tn"]],
  [ATTN_WGSL, ["attn_fwd", "attn_dv", "attn_dsq", "attn_dk"]],
  [LN_WGSL, ["ln_fwd", "ln_dx", "ln_dgb"]],
  [MISC_WGSL, ["embed_fwd", "embed_dwte", "embed_dwpe", "addv", "gelu_fwd", "gelu_bwd", "ce", "adam"]],
]) {
  const mod = device.createShaderModule({ code: src });
  for (const n of names) {
    pipelines[n] = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: n } });
  }
}

const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const buf = (n) => device.createBuffer({ size: Math.max(n * 4, 16), usage: SU });
const ubuf = (arr) => {
  const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, arr instanceof Uint32Array ? arr : new Uint32Array(arr));
  return b;
};

// ---------- 参数张量（顺序与 CPU params() 严格一致） ----------
function tensorList() {
  const L = [];
  const add = (name, rows, cols) => L.push({ name, rows, cols, n: rows * cols });
  add("wte", V, E); add("wpe", CFG.blockSize, E);
  add("lnFg", 1, E); add("lnFb", 1, E);
  for (let l = 0; l < CFG.nLayer; l++) {
    add(`b${l}.ln1g`, 1, E); add(`b${l}.ln1b`, 1, E);
    add(`b${l}.Wq`, E, E); add(`b${l}.Wk`, E, E); add(`b${l}.Wv`, E, E); add(`b${l}.Wo`, E, E);
    add(`b${l}.ln2g`, 1, E); add(`b${l}.ln2b`, 1, E);
    add(`b${l}.W1`, E, FF); add(`b${l}.W2`, FF, E);
  }
  return L;
}
const params = tensorList();
for (const p of params) { p.w = buf(p.n); p.g = buf(p.n); p.m = buf(p.n); p.v = buf(p.n); }
const P = Object.fromEntries(params.map((p) => [p.name, p]));
const totalParams = params.reduce((s, p) => s + p.n, 0);

// ---------- 权重加载/保存（二进制 meta+bin，与 CPU 互通；JSON 在 5000 万参数下超 V8 字符串上限）----------
function loadWeightsToGPU() {
  const meta = JSON.parse(Deno.readTextFileSync(META));
  const c = meta.cfg;
  if (c.vocabSize !== V || c.nLayer !== CFG.nLayer || c.nEmbd !== E) {
    console.error("权重与模型结构不匹配"); Deno.exit(1);
  }
  const raw = Deno.readFileSync(BIN);
  let off = raw.byteOffset;
  params.forEach((p) => {
    device.queue.writeBuffer(p.w, 0, raw.buffer, off, p.n * 4);
    off += p.n * 4;
  });
  return meta;
}
function randnFlat(n, std = 0.05) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const u = 1 - Math.random(), v = Math.random();
    a[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
  }
  return a;
}
function initRandom() {
  for (const p of params) {
    if (p.name.includes("ln") && p.name.endsWith("g")) {
      device.queue.writeBuffer(p.w, 0, new Float32Array(p.n).fill(1));
    } else if (p.name.includes("ln")) {
      device.queue.writeBuffer(p.w, 0, new Float32Array(p.n));
    } else {
      device.queue.writeBuffer(p.w, 0, randnFlat(p.n));
    }
  }
}
async function readBuf(b, n) {
  const st = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(b, 0, st, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(st.getMappedRange().slice(0));
  st.unmap(); st.destroy();
  return out;
}
async function saveWeights(step, totalSteps, extra = null, bin = BIN, meta_ = META) {
  const all = new Float32Array(totalParams);
  let off = 0;
  for (const p of params) {
    const flat = await readBuf(p.w, p.n);
    for (const x of flat) {
      if (!Number.isFinite(x)) {   // 拒绝毒权重上盘，保护旧文件
        console.error(`[saveWeights] ${p.name} 含非法值，拒绝存盘！`);
        return false;
      }
    }
    all.set(flat, off); off += p.n;
  }
  const meta = {
    format: "bin-f32",
    cfg: { vocabSize: V, nLayer: CFG.nLayer, nHead: H, nEmbd: E, blockSize: CFG.blockSize },
    step, totalSteps,
    // 字表必须随模型走。以前推理端是从语料现算字表，一旦语料换了
    // （比如加入宋诗），旧权重的字表就对不上，输出变乱码而不报错。
    vocab: chars,
    corpus: CORPUS_NAME,
    tensors: params.map((p) => ({ name: p.name, rows: p.rows, cols: p.cols })),
    ...(extra || {}),
  };
  Deno.writeFileSync(bin + ".tmp", new Uint8Array(all.buffer));
  Deno.renameSync(bin + ".tmp", bin);
  Deno.writeTextFileSync(meta_ + ".tmp", JSON.stringify(meta));
  Deno.renameSync(meta_ + ".tmp", meta_);
  return true;
}

// ---------- 激活缓冲 ----------
const act = { X0: buf(BT * E), ids: null, tgt: null, loss: buf(BT), logits: buf(BT * V), dLogits: buf(BT * V), hF: buf(BT * E), xhF: buf(BT * E), isF: buf(BT) };
act.ids = device.createBuffer({ size: BT * 4, usage: SU });
act.tgt = device.createBuffer({ size: BT * 4, usage: SU });
const layers = [];
for (let l = 0; l < CFG.nLayer; l++) {
  layers.push({
    h1: buf(BT * E), xh1: buf(BT * E), is1: buf(BT),
    q: buf(BT * E), k: buf(BT * E), v: buf(BT * E),
    aw: buf(B * H * T * T), ao: buf(BT * E), proj: buf(BT * E), Xa: buf(BT * E),
    h2: buf(BT * E), xh2: buf(BT * E), is2: buf(BT),
    m1: buf(BT * FF), g1: buf(BT * FF), m2: buf(BT * E), Xb: buf(BT * E),
  });
}
// 反向缓冲（跨层复用）
const bwd = {
  gXb: buf(BT * E), gXa: buf(BT * E), gXprev: buf(BT * E),
  gG1: buf(BT * FF), gM1: buf(BT * FF), gH2: buf(BT * E),
  gAO: buf(BT * E), gQ: buf(BT * E), gK: buf(BT * E), gV: buf(BT * E),
  dS: buf(B * H * T * T), gH1: buf(BT * E), gHF: buf(BT * E),
};
const adamPB = buf(3);   // [lr, c1, c2]

// ---------- bind group 缓存 ----------
let bufIdSeq = 0;
const bufIds = new WeakMap();
const bid = (b) => { if (!bufIds.has(b)) bufIds.set(b, ++bufIdSeq); return bufIds.get(b); };
const bgCache = new Map();
const uCache = new Map();
function uniformFor(key, arr) {
  if (!uCache.has(key)) uCache.set(key, ubuf(arr));
  return uCache.get(key);
}
// 每个内核实际引用的绑定点（layout:"auto" 会剔除未用绑定，bind group 必须同步过滤）
const BIND_MASK = {
  mm_nn: [0, 1, 2, 3], mm_nt: [0, 1, 2, 3], mm_tn: [0, 1, 2, 3],
  attn_fwd: [0, 1, 2, 3, 4, 5], attn_dv: [0, 1, 2, 4],
  attn_dsq: [0, 1, 2, 3, 4, 5, 6], attn_dk: [0, 1, 2, 4],
  ln_fwd: [0, 1, 2, 3, 4, 5, 6], ln_dx: [0, 1, 2, 4, 5, 6], ln_dgb: [0, 1, 4, 5, 6],
  embed_fwd: [0, 1, 2, 3, 4], embed_dwte: [0, 1, 2, 4], embed_dwpe: [0, 2, 4],
  addv: [0, 2, 3, 4], gelu_fwd: [0, 2, 4], gelu_bwd: [0, 2, 3, 4],
  ce: [0, 1, 2, 4, 5], adam: [0, 3, 4, 5, 6, 7],
};

function bind(name, entries) {
  const pipe = pipelines[name];
  const mask = BIND_MASK[name];
  const key = name + "|" + mask.map((i) => bid(entries[i])).join(",");
  if (!bgCache.has(key)) {
    bgCache.set(key, device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: mask.map((i) => ({ binding: i, resource: { buffer: entries[i] } })),
    }));
  }
  return bgCache.get(key);
}
// (label 已不需要：bind 改为显式传内核名)

// ---------- 派发助手 ----------
function mm(pass, kind, A, Bm, C, n, k, m, accum) {
  const u = uniformFor(`mm|${n},${k},${m},${accum}`, [n, k, m, accum]);
  pass.setPipeline(pipelines[kind]);
  pass.setBindGroup(0, bind(kind, [u, A, Bm, C]));
  pass.dispatchWorkgroups(Math.ceil(m / 64), Math.ceil(n / 64));
}
const attnU = uniformFor("attn", [B, H, T, E]);
function attn(pass, kind, bufs) {
  pass.setPipeline(pipelines[kind]);
  pass.setBindGroup(0, bind(kind, [attnU, ...bufs]));
  pass.dispatchWorkgroups(Math.ceil((B * H * T) / 64));
}
function ln(pass, kind, rows, cols, accum, bufs) {
  const u = uniformFor(`ln|${rows},${cols},${accum}`, [rows, cols, accum, 0]);
  pass.setPipeline(pipelines[kind]);
  pass.setBindGroup(0, bind(kind, [u, ...bufs]));
  const thr = kind === "ln_dgb" ? cols : rows;
  pass.dispatchWorkgroups(Math.ceil(thr / 64));
}
const dummyU32 = device.createBuffer({ size: 16, usage: SU });
const dummyF = device.createBuffer({ size: 16, usage: SU });
// ce 专用 uniform（有效位置数逐 batch 变化，不进缓存）
const ceU = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
function misc(pass, kind, dims, bufs, threads, wg = 256, uOverride = null) {
  const u = uOverride || uniformFor(`misc|${kind}|${dims.join(",")}`, [...dims, 0, 0, 0].slice(0, 4));
  pass.setPipeline(pipelines[kind]);
  pass.setBindGroup(0, bind(kind, [u, ...bufs]));
  pass.dispatchWorkgroups(Math.ceil(threads / wg));
}

// ---------- 单步：前向 + 反向 + （可选）Adam ----------
function encodeStep(enc, withAdam) {
  // 清参数梯度
  for (const p of params) enc.clearBuffer(p.g);
  const pass = enc.beginComputePass();
  // ---- 前向 ----
  misc(pass, "embed_fwd", [BT, T, E], [act.ids, P.wte.w, P.wpe.w, act.X0, dummyF], BT * E);
  let Xin = act.X0;
  for (let l = 0; l < CFG.nLayer; l++) {
    const L = layers[l], pb = (s) => P[`b${l}.${s}`].w;
    ln(pass, "ln_fwd", BT, E, 0, [Xin, pb("ln1g"), pb("ln1b"), L.h1, L.xh1, L.is1]);
    mm(pass, "mm_nn", L.h1, pb("Wq"), L.q, BT, E, E, 0);
    mm(pass, "mm_nn", L.h1, pb("Wk"), L.k, BT, E, E, 0);
    mm(pass, "mm_nn", L.h1, pb("Wv"), L.v, BT, E, E, 0);
    attn(pass, "attn_fwd", [L.q, L.k, L.v, L.aw, L.ao, dummyF]);
    mm(pass, "mm_nn", L.ao, pb("Wo"), L.proj, BT, E, E, 0);
    misc(pass, "addv", [BT * E], [dummyU32, Xin, L.proj, L.Xa, dummyF], BT * E);
    ln(pass, "ln_fwd", BT, E, 0, [L.Xa, pb("ln2g"), pb("ln2b"), L.h2, L.xh2, L.is2]);
    mm(pass, "mm_nn", L.h2, pb("W1"), L.m1, BT, E, FF, 0);
    misc(pass, "gelu_fwd", [BT * FF], [dummyU32, L.m1, dummyF, L.g1, dummyF], BT * FF);
    mm(pass, "mm_nn", L.g1, pb("W2"), L.m2, BT, FF, E, 0);
    misc(pass, "addv", [BT * E], [dummyU32, L.Xa, L.m2, L.Xb, dummyF], BT * E);
    Xin = L.Xb;
  }
  ln(pass, "ln_fwd", BT, E, 0, [Xin, P.lnFg.w, P.lnFb.w, act.hF, act.xhF, act.isF]);
  mm(pass, "mm_nt", act.hF, P.wte.w, act.logits, BT, E, V, 0);
  // ---- 损失 + dLogits ----
  misc(pass, "ce", [BT, V], [act.tgt, act.logits, dummyF, act.dLogits, act.loss], BT, 64, ceU);
  // ---- 反向 ----
  mm(pass, "mm_tn", act.dLogits, act.hF, P.wte.g, V, BT, E, 1);       // dwte += dLogitsᵀ·hF
  mm(pass, "mm_nn", act.dLogits, P.wte.w, bwd.gHF, BT, V, E, 0);      // dhF = dLogits·wte
  ln(pass, "ln_dx", BT, E, 0, [bwd.gHF, P.lnFg.w, dummyF, bwd.gXb, act.xhF, act.isF]);
  ln(pass, "ln_dgb", BT, E, 0, [bwd.gHF, dummyF, dummyF, P.lnFg.g, act.xhF, P.lnFb.g]);
  pass.end();

  for (let l = CFG.nLayer - 1; l >= 0; l--) {
    const L = layers[l], pw = (s) => P[`b${l}.${s}`].w, pg = (s) => P[`b${l}.${s}`].g;
    const XinL = l === 0 ? act.X0 : layers[l - 1].Xb;
    // 残差直通：gXa = gXb（copy 在 pass 外）
    enc.copyBufferToBuffer(bwd.gXb, 0, bwd.gXa, 0, BT * E * 4);
    const pass2 = enc.beginComputePass();
    // MLP 支路
    mm(pass2, "mm_nt", bwd.gXb, pw("W2"), bwd.gG1, BT, E, FF, 0);
    mm(pass2, "mm_tn", L.g1, bwd.gXb, pg("W2"), FF, BT, E, 1);
    misc(pass2, "gelu_bwd", [BT * FF], [dummyU32, L.m1, bwd.gG1, bwd.gM1, dummyF], BT * FF);
    mm(pass2, "mm_nt", bwd.gM1, pw("W1"), bwd.gH2, BT, FF, E, 0);
    mm(pass2, "mm_tn", L.h2, bwd.gM1, pg("W1"), E, BT, FF, 1);
    ln(pass2, "ln_dx", BT, E, 1, [bwd.gH2, pw("ln2g"), dummyF, bwd.gXa, L.xh2, L.is2]);
    ln(pass2, "ln_dgb", BT, E, 0, [bwd.gH2, dummyF, dummyF, pg("ln2g"), L.xh2, pg("ln2b")]);
    // 注意力支路
    mm(pass2, "mm_nt", bwd.gXa, pw("Wo"), bwd.gAO, BT, E, E, 0);
    mm(pass2, "mm_tn", L.ao, bwd.gXa, pg("Wo"), E, BT, E, 1);
    attn(pass2, "attn_dv", [L.aw, bwd.gAO, dummyF, bwd.gV, dummyF]);
    attn(pass2, "attn_dsq", [L.aw, bwd.gAO, L.v, bwd.dS, bwd.gQ, L.k]);
    attn(pass2, "attn_dk", [bwd.dS, L.q, dummyF, bwd.gK, dummyF]);
    mm(pass2, "mm_nt", bwd.gQ, pw("Wq"), bwd.gH1, BT, E, E, 0);
    mm(pass2, "mm_nt", bwd.gK, pw("Wk"), bwd.gH1, BT, E, E, 1);
    mm(pass2, "mm_nt", bwd.gV, pw("Wv"), bwd.gH1, BT, E, E, 1);
    mm(pass2, "mm_tn", L.h1, bwd.gQ, pg("Wq"), E, BT, E, 1);
    mm(pass2, "mm_tn", L.h1, bwd.gK, pg("Wk"), E, BT, E, 1);
    mm(pass2, "mm_tn", L.h1, bwd.gV, pg("Wv"), E, BT, E, 1);
    pass2.end();
    // gXprev = gXa（残差）+ ln1_dx(gH1)
    enc.copyBufferToBuffer(bwd.gXa, 0, bwd.gXprev, 0, BT * E * 4);
    const pass3 = enc.beginComputePass();
    ln(pass3, "ln_dx", BT, E, 1, [bwd.gH1, pw("ln1g"), dummyF, bwd.gXprev, L.xh1, L.is1]);
    ln(pass3, "ln_dgb", BT, E, 0, [bwd.gH1, dummyF, dummyF, pg("ln1g"), L.xh1, pg("ln1b")]);
    pass3.end();
    enc.copyBufferToBuffer(bwd.gXprev, 0, bwd.gXb, 0, BT * E * 4);
  }
  const pass4 = enc.beginComputePass();
  misc(pass4, "embed_dwte", [BT, V, E], [act.ids, bwd.gXb, dummyF, P.wte.g, dummyF], V * E);
  misc(pass4, "embed_dwpe", [B, T, E], [act.ids, bwd.gXb, dummyF, P.wpe.g, dummyF], T * E);
  // Adam
  if (withAdam) {
    for (const p of params) {
      misc(pass4, "adam", [p.n], [dummyU32, dummyF, adamPB, p.w, p.g, p.m, p.v], p.n);
    }
  }
  pass4.end();
}

// ---------- batch 上传（变长诗：末尾 padding，目标位用哨兵值掩码）----------
let lastRealCount = BT;
function uploadBatch(poemIdxs) {
  const xs = new Uint32Array(BT);           // 输入 pad 用 0（\n），因果掩码不影响损失
  const ys = new Uint32Array(BT).fill(PAD); // 目标 pad 用哨兵值
  let real = 0;
  poemIdxs.forEach((pi, b) => {
    if (pi == null) return;                 // 空槽：全 padding，不贡献损失
    const ids = encode("\n" + POEMS[pi] + "\n");
    const L = ids.length - 1;               // 本诗有效预测位置数
    for (let t = 0; t < L; t++) { xs[b * T + t] = ids[t]; ys[b * T + t] = ids[t + 1]; }
    real += L;
  });
  lastRealCount = real;
  device.queue.writeBuffer(act.ids, 0, xs);
  device.queue.writeBuffer(act.tgt, 0, ys);
  device.queue.writeBuffer(ceU, 0, new Uint32Array([BT, V, real, 0]));
}

// ---------- 验证集评估：跑完 704 首留出诗，返回每 token 平均损失 ----------
// 复用 encodeStep(enc, false)：会算反向但不跑 Adam，梯度会在下一个训练步开头被清掉，
// 权重与 Adam 状态一律不受影响。多算的反向只是浪费一点时间，换零改动风险。
async function evalVal() {
  let lossSum = 0, tokSum = 0;
  for (let i = 0; i < VAL.length; i += B) {
    uploadBatch(VAL.slice(i, i + B));
    const enc = device.createCommandEncoder();
    encodeStep(enc, false);
    device.queue.submit([enc.finish()]);
    const arr = await readBuf(act.loss, BT);
    for (const x of arr) lossSum += x;
    tokSum += lastRealCount;
  }
  return lossSum / tokSum;
}

// ============================================================
// 主流程
// ============================================================

console.log(`GPU 训练器 | ${CFG.nLayer}层/${H}头/${E}维 | 参数 ${totalParams.toLocaleString()} | batch=${B}`);
console.log(`数据切分 | 训练 ${TRAIN.length} 首 / 验证 ${VAL.length} 首（留出，不参与训练）| 权重前缀 ${PREFIX}`);

let startStep = 0;
let totalSteps = Number(Deno.args[0]) || 300000;
let haveWeights = false;
try { Deno.statSync(META); Deno.statSync(BIN); haveWeights = true; } catch { /* 无二进制权重 */ }
if (haveWeights) {
  console.log("加载存盘权重...");
  const meta = loadWeightsToGPU();
  startStep = meta.step || 0;
  if (!Deno.args[0] && meta.totalSteps) totalSteps = meta.totalSteps;
  console.log(`续训: ${startStep} → ${totalSteps}`);
} else {
  console.log("随机初始化，从零训练");
  initRandom();
}

// ---------- 自检：32 份同一首诗 vs CPU 单首（loss 应一致） ----------
{
  console.log("\n自检：GPU forward/backward vs CPU（同权重同数据）...");
  uploadBatch(new Array(B).fill(0));
  const enc = device.createCommandEncoder();
  encodeStep(enc, false);   // 只算梯度不更新
  device.queue.submit([enc.finish()]);
  const lossArr = await readBuf(act.loss, BT);
  const gpuLoss = lossArr.reduce((s, x) => s + x, 0) / lastRealCount;

  // CPU 参照
  // 只在语料仍是 poems.js 时做：CPU 引擎的字表是它自己从 poems.js 算的，
  // 换了大语料后我们的 token id 最大可到 9063，喂给 6379 行的 wte 会越界。
  let cpuLoss = NaN, cpuOk = CORPUS_NAME === "poems.js";
  if (!cpuOk) {
    console.log(`  语料为 ${CORPUS_NAME}，字表与 CPU 引擎不同，跳过 CPU 对拍`);
  } else {
    const cpu = require("./mini-gpt-poet.js");
    const cpuModel = new cpu.MiniGPT(cpu.CFG);
    if (haveWeights) cpu.loadWeights(cpuModel);
    const ids0 = encode("\n" + POEMS[0] + "\n");
    const cpuLossT = cpu.crossEntropy(cpuModel.forward(ids0.slice(0, -1)), ids0.slice(1));
    cpuLoss = cpuLossT.data[0][0];
    console.log(`  loss: GPU=${gpuLoss.toFixed(5)}  CPU=${cpuLoss.toFixed(5)}  差=${Math.abs(gpuLoss - cpuLoss).toExponential(2)}`);
    if (haveWeights) {
      cpu.backward(cpuLossT);
      const cpuWpe = cpuModel.params()[1];   // wpe
      const gpuWpeG = await readBuf(P.wpe.g, P.wpe.n);
      let maxDiff = 0;
      for (let i = 0; i < T; i++)
        for (let j = 0; j < E; j++)
          maxDiff = Math.max(maxDiff, Math.abs(gpuWpeG[i * E + j] - cpuWpe.grad[i][j]));
      console.log(`  wpe 梯度最大误差: ${maxDiff.toExponential(2)}`);
      if (!Number.isFinite(gpuLoss) || !Number.isFinite(maxDiff) ||
          Math.abs(gpuLoss - cpuLoss) > 5e-3 || maxDiff > 5e-3) {
        if (Deno.env.get("DEBUG")) await debugDump();
        console.error("自检失败，中止！"); Deno.exit(1);
      }
    }
  }
  if (!haveWeights) {
    // 随机初始化理论 loss ≈ ln(V) + 随机 logits 方差项(~0.5)，宽容到 ±1.2。
    // 这条判据只对随机初始化成立：续训时模型已训练，loss 远低于 ln(V)，
    // 拿它来卡会把正常的续训误判为异常并中止。
    console.log(`  随机初始 loss ${gpuLoss.toFixed(5)}  理论 ln(V)=${Math.log(V).toFixed(5)}`);
    if (!Number.isFinite(gpuLoss) || Math.abs(gpuLoss - Math.log(V)) > 1.2) {
      if (Deno.env.get("DEBUG")) await debugDump();
      console.error("随机初始化 loss 异常，中止！"); Deno.exit(1);
    }
  } else if (!cpuOk) {
    // 续训且无 CPU 参照：只能验 loss 是有限值，以及量级合理（不该高于 ln(V)）
    console.log(`  续训自检 loss ${gpuLoss.toFixed(5)}（ln(V)=${Math.log(V).toFixed(5)}，应明显更低）`);
    if (!Number.isFinite(gpuLoss) || gpuLoss > Math.log(V) + 0.5) {
      if (Deno.env.get("DEBUG")) await debugDump();
      console.error("续训 loss 异常（权重可能损坏），中止！"); Deno.exit(1);
    }
  }
  console.log("自检通过 ✅\n");
}

// 调试：逐缓冲区扫 NaN，定位毒源
async function debugDump() {
  const probe = async (name, b, n) => {
    const a = await readBuf(b, Math.min(n, 999999));
    let nan = 0, mx = 0;
    for (const x of a) { if (!Number.isFinite(x)) nan++; else mx = Math.max(mx, Math.abs(x)); }
    console.log(`  [dbg] ${name}: nan=${nan}/${a.length}  maxAbs=${mx.toFixed(4)}`);
    return nan;
  };
  await probe("X0", act.X0, BT * E);
  for (let l = 0; l < CFG.nLayer; l++) {
    const L = layers[l];
    const bad =
      (await probe(`L${l}.h1`, L.h1, BT * E)) ||
      (await probe(`L${l}.q`, L.q, BT * E)) ||
      (await probe(`L${l}.aw`, L.aw, B * H * T * T)) ||
      (await probe(`L${l}.ao`, L.ao, BT * E)) ||
      (await probe(`L${l}.Xa`, L.Xa, BT * E)) ||
      (await probe(`L${l}.m1`, L.m1, BT * FF)) ||
      (await probe(`L${l}.Xb`, L.Xb, BT * E));
    if (bad) { console.log(`  → 毒源在第 ${l} 层`); break; }
  }
  await probe("hF", act.hF, BT * E);
  await probe("logits", act.logits, BT * V);
}

// ---------- 训练循环 ----------
if (startStep >= totalSteps) {
  console.log("已达目标步数，无需训练");
} else {
  console.log(`训练 ${startStep} → ${totalSteps}（每步 ${B} 样本）...`);
  const t0 = performance.now();
  let step = startStep, adamT = 0, lossSum = 0, lossN = 0;
  let nextLog = Math.ceil((step + 1) / 5000) * 5000;
  let nextCkpt = Math.ceil((step + 1) / 50000) * 50000;   // 二进制存盘快，加密 checkpoint
  let nextEval = Math.ceil((step + 1) / EVAL_EVERY) * EVAL_EVERY;
  let bestVal = Infinity, bestStep = 0, trainRecent = NaN;
  try { Deno.statSync(CURVE); } catch { Deno.writeTextFileSync(CURVE, "step,train,val\n"); }

  while (step < totalSteps) {
    // lr 全程恒定：上一炉在 60% 处降到 1/3，实测 train loss 从 2.32 骤降到 1.34
    // 而 val loss 同步上翻——降学习率把模型直接推进了背书阶段，是有害的。
    const lr = BASE_LR;
    adamT++;
    device.queue.writeBuffer(adamPB, 0, new Float32Array([lr, 1 - 0.9 ** adamT, 1 - 0.999 ** adamT]));
    uploadBatch(Array.from({ length: B }, () => TRAIN[Math.floor(Math.random() * TRAIN.length)]));
    const enc = device.createCommandEncoder();
    encodeStep(enc, true);
    device.queue.submit([enc.finish()]);
    step += B;

    if (step >= nextLog) {
      const lossArr = await readBuf(act.loss, BT);   // 顺带同步一次
      const cur = lossArr.reduce((s, x) => s + x, 0) / lastRealCount;
      if (!Number.isFinite(cur)) {
        console.error(`step ${step}: loss=NaN，训练发散，立即中止（不存盘）`);
        Deno.exit(1);
      }
      lossSum += cur; lossN++;
      trainRecent = lossSum / lossN;
      const speed = (performance.now() - t0) / (step - startStep);
      console.log(`  step ${step}  loss = ${trainRecent.toFixed(4)}  (${speed.toFixed(3)}ms/样本)`);
      lossSum = 0; lossN = 0; nextLog += 5000;
    }
    if (step >= nextEval) {
      // 若还没到过 5000 步日志点，trainRecent 仍是 NaN，用当前 batch 现算一个。
      // 必须在 evalVal() 之前读，否则 act.loss 已被验证集覆盖。
      if (!Number.isFinite(trainRecent)) {
        const a = await readBuf(act.loss, BT);
        trainRecent = a.reduce((s, x) => s + x, 0) / lastRealCount;
      }
      const val = await evalVal();
      const gap = val - trainRecent;
      Deno.writeTextFileSync(CURVE, `${step},${trainRecent.toFixed(4)},${val.toFixed(4)}\n`, { append: true });
      let mark = "";
      const extra = { valLoss: val, trainLoss: trainRecent };
      if (val < bestVal) {
        bestVal = val; bestStep = step;
        await saveWeights(step, totalSteps, extra,
          BIN.replace(/\.bin$/, "-best.bin"), META.replace(/\.meta\.json$/, "-best.meta.json"));
        mark = "  ← 新低，已存 best";
      }
      // 平台期候选存档：val 在最优值 5% 以内就留一份带步数的快照。
      // 上一炉只存了「新低」，而 20 万/22.5 万步那两个候选恰好不是新低
      // （val 4.537/4.571 vs 最优 4.511），统计上几乎同样好但多训了几万步、
      // 词汇更丰富，结果一份没留下来。真正的甜点在平台上，不在极小值点。
      if (val < bestVal * 1.05) {
        await saveWeights(step, totalSteps, extra,
          BIN.replace(/\.bin$/, `-v${step}.bin`), META.replace(/\.meta\.json$/, `-v${step}.meta.json`));
        mark += `  [存档 v${step}]`;
      }
      console.log(`  [验证] step ${step}  train ${trainRecent.toFixed(4)}  val ${val.toFixed(4)}  gap ${gap >= 0 ? "+" : ""}${gap.toFixed(4)}${mark}`);
      nextEval += EVAL_EVERY;
    }
    if (step >= nextCkpt) {
      await device.queue.onSubmittedWorkDone();
      await saveWeights(step, totalSteps);
      console.log(`  [checkpoint] 已存盘 @ step ${step}`);
      nextCkpt += 50000;
    }
  }
  await device.queue.onSubmittedWorkDone();
  await saveWeights(totalSteps, totalSteps);
  const hrs = (performance.now() - t0) / 3600000;
  console.log(`\n训练完成，耗时 ${(hrs * 60).toFixed(1)} 分钟`);
  console.log(`验证集最优: step ${bestStep}  val loss ${bestVal.toFixed(4)}  →  ${PREFIX}-best.bin`);
  console.log(`双曲线数据: ${PREFIX}-curve.csv`);
}
