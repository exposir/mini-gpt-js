// ============================================================
// webgpu-forward.js —— GPU 推理模块（前向专用，环境无关 ES Module）
//
// 同一份代码跑在两个地方：
//   - Deno:   gpu-server.js 导入，做服务端 GPU 推理
//   - 浏览器: 展示页 <script type="module"> 导入，页内本地推理
//
// 用法:
//   const poet = await createPoet(meta, binBuffer, chars);
//   const poem = await poet.generate("月", { temperature: 0.6, topK: 5 });
// ============================================================

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
      let ar = idx / 16u; let ac = idx % 16u;
      let aRow = wid.y * 64u + ar; let aCol = t * 16u + ac;
      var av = 0.0;
      if (aRow < dims.n && aCol < dims.k) { av = A[aRow * dims.k + aCol]; }
      tA[idx] = av;
      let br = idx / 64u; let bc = idx % 64u;
      let bRow = t * 16u + br; let bCol = wid.x * 64u + bc;
      var bv = 0.0;
      if (bRow < dims.k && bCol < dims.m) {
        if (mode == 1u) { bv = B[bCol * dims.k + bRow]; }
        else            { bv = B[bRow * dims.m + bCol]; }
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
      if (col < dims.m) { C[row * dims.m + col] = acc[r * 4u + c]; }
    }
  }
}
@compute @workgroup_size(16,16) fn mm_nn(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) { body(w, l, 0u); }
@compute @workgroup_size(16,16) fn mm_nt(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) { body(w, l, 1u); }
`;

const OPS_WGSL = /* wgsl */ `
struct D { a: u32, b: u32, c: u32, d2: u32 };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> IU: array<u32>;
@group(0) @binding(2) var<storage, read> F1: array<f32>;
@group(0) @binding(3) var<storage, read> F2: array<f32>;
@group(0) @binding(4) var<storage, read_write> O1: array<f32>;

// embed: a=T b=blockSize c=E | IU=ids F1=wte F2=wpe O1=X
@compute @workgroup_size(256) fn embed_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= d.a * d.c) { return; }
  let p = idx / d.c; let e = idx % d.c;
  O1[idx] = F1[IU[p]*d.c+e] + F2[(p % d.b)*d.c+e];
}
// ln: a=rows b=cols | F1=X F2=g(前半)b(后半拼接) O1=Y   —— g、b 拼进一个缓冲
@compute @workgroup_size(64) fn ln_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= d.a) { return; }
  var mean = 0.0;
  for (var j = 0u; j < d.b; j++) { mean += F1[i*d.b+j]; }
  mean /= f32(d.b);
  var va = 0.0;
  for (var j = 0u; j < d.b; j++) { let t = F1[i*d.b+j]-mean; va += t*t; }
  let inv = 1.0 / sqrt(va / f32(d.b) + 1e-5);
  for (var j = 0u; j < d.b; j++) {
    O1[i*d.b+j] = (F1[i*d.b+j] - mean) * inv * F2[j] + F2[d.b + j];
  }
}
// attn: a=T b=nHead c=E | F1=QKV拼接(3段) O1=out
@compute @workgroup_size(64) fn attn_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
  let hd = d.c / d.b;
  let total = d.b * d.a;
  if (g.x >= total) { return; }
  let i = g.x % d.a; let h = g.x / d.a;
  let off = h * hd;
  let qBase = 0u; let kBase = d.a * d.c; let vBase = 2u * d.a * d.c;
  let scale = 1.0 / sqrt(f32(hd));
  var scores: array<f32, 72>;
  var mx = -1e30;
  for (var j = 0u; j <= i; j++) {
    var dot = 0.0;
    for (var k = 0u; k < hd; k++) {
      dot += F1[qBase + i*d.c+off+k] * F1[kBase + j*d.c+off+k];
    }
    scores[j] = dot * scale;
    mx = max(mx, scores[j]);
  }
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) { scores[j] = exp(scores[j] - mx); sum += scores[j]; }
  for (var k = 0u; k < hd; k++) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc += (scores[j] / sum) * F1[vBase + j*d.c+off+k]; }
    O1[i*d.c+off+k] = acc;
  }
}
// add: a=count | F1+F2 → O1
@compute @workgroup_size(256) fn addv(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= d.a) { return; }
  O1[g.x] = F1[g.x] + F2[g.x];
}
// gelu: a=count | F1 → O1
@compute @workgroup_size(256) fn gelu_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= d.a) { return; }
  let x = F1[g.x];
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x*x*x), -9.0, 9.0);
  O1[g.x] = 0.5 * x * (1.0 + tanh(t));
}
`;

const BIND_MASK = {
  mm_nn: [0, 1, 2, 3], mm_nt: [0, 1, 2, 3],
  embed_fwd: [0, 1, 2, 3, 4], ln_fwd: [0, 2, 3, 4],
  attn_fwd: [0, 2, 4], addv: [0, 2, 3, 4], gelu_fwd: [0, 2, 4],
  // KV Cache 单 token 路径
  embed_step: [0, 1, 2, 3, 4], ln_s: [0, 2, 3, 4], mv: [0, 2, 3, 4], mvT: [0, 2, 3, 4],
  attn_step: [0, 2, 3, 4, 5], addv_s: [0, 2, 3, 4], gelu_s: [0, 2, 4], sample: [0, 2, 3, 6],
};

// ============================================================
// 单 token 内核（KV Cache + 批处理）
//   激活按 [batch][维度] 紧凑排列，K/V cache 按 [batch][pos][E]
//   权重不带 batch 维（所有序列共享）—— 派发数与 batch 无关，
//   这是批处理能近乎免费的原因：瓶颈是派发开销，不是算术
// 绑定布局: 0=uniform 1=IU(ids) 2=F1 3=F2 4=O1(rw) 5=F3 6=OU(ids 可写)
// ============================================================
const STEP_WGSL = /* wgsl */ `
struct D { a: u32, b: u32, c: u32, d2: u32 };
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> IU: array<u32>;
@group(0) @binding(2) var<storage, read> F1: array<f32>;
@group(0) @binding(3) var<storage, read> F2: array<f32>;
@group(0) @binding(4) var<storage, read_write> O1: array<f32>;
@group(0) @binding(5) var<storage, read> F3: array<f32>;
@group(0) @binding(6) var<storage, read_write> OU: array<u32>;

// embed_step: a=E b=T(ids步长) c=pos | IU=ids F1=wte F2=wpe → O1=x[batch][E]
@compute @workgroup_size(64) fn embed_step(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  let bi = idx / d.a; let e = idx % d.a;
  if (e >= d.a) { return; }
  O1[idx] = F1[IU[bi * d.b + d.c] * d.a + e] + F2[d.c * d.a + e];
}

// ln_s: a=cols | 每个 workgroup 处理一个 batch 项的一行
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64) fn ln_s(@builtin(workgroup_id) wid: vec3<u32>,
                                     @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = d.a;
  let base = wid.x * n;
  var s = 0.0;
  for (var j = lid.x; j < n; j += 64u) { s += F1[base + j]; }
  red[lid.x] = s;
  workgroupBarrier();
  if (lid.x == 0u) { var t = 0.0; for (var i = 0u; i < 64u; i++) { t += red[i]; } red[0] = t / f32(n); }
  workgroupBarrier();
  let mean = red[0];
  workgroupBarrier();
  var v = 0.0;
  for (var j = lid.x; j < n; j += 64u) { let t = F1[base + j] - mean; v += t * t; }
  red[lid.x] = v;
  workgroupBarrier();
  if (lid.x == 0u) { var t = 0.0; for (var i = 0u; i < 64u; i++) { t += red[i]; } red[0] = t / f32(n); }
  workgroupBarrier();
  let inv = 1.0 / sqrt(red[0] + 1e-5);
  for (var j = lid.x; j < n; j += 64u) { O1[base + j] = (F1[base + j] - mean) * inv * F2[j] + F2[n + j]; }
}

// mv: a=k b=m c=行偏移 d2=行步长 | F1=vec[batch][k] F2=Mat(k×m) → O1[(bi*d2+c)*m + j]
//     d2=1,c=0 → 普通激活；d2=T,c=pos → 写入 K/V cache 的 pos 行
@compute @workgroup_size(64) fn mv(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  let bi = idx / d.b; let j = idx % d.b;
  var acc = 0.0;
  let ib = bi * d.a;
  for (var p = 0u; p < d.a; p++) { acc += F1[ib + p] * F2[p * d.b + j]; }
  O1[(bi * d.d2 + d.c) * d.b + j] = acc;
}

// mvT: a=k b=m | F1=vec[batch][k] F2=Mat(m×k 行向量) → O1[batch][m]
@compute @workgroup_size(64) fn mvT(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  let bi = idx / d.b; let j = idx % d.b;
  var acc = 0.0;
  let ib = bi * d.a;
  for (var p = 0u; p < d.a; p++) { acc += F1[ib + p] * F2[j * d.a + p]; }
  O1[bi * d.b + j] = acc;
}

// attn_step: a=有效长度 b=nHead c=E d2=T | F1=q[batch][E] F2=Kcache F3=Vcache → O1=out[batch][E]
@compute @workgroup_size(64) fn attn_step(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  let bi = idx / d.b; let h = idx % d.b;
  let hd = d.c / d.b;
  let off = h * hd;
  let qb = bi * d.c + off;
  let cb = bi * d.d2 * d.c;          // 本 batch 项的 cache 起点
  let scale = 1.0 / sqrt(f32(hd));
  var scores: array<f32, 72>;
  var mx = -1e30;
  for (var j = 0u; j < d.a; j++) {
    var dot = 0.0;
    for (var k = 0u; k < hd; k++) { dot += F1[qb + k] * F2[cb + j * d.c + off + k]; }
    scores[j] = dot * scale;
    mx = max(mx, scores[j]);
  }
  var sum = 0.0;
  for (var j = 0u; j < d.a; j++) { scores[j] = exp(scores[j] - mx); sum += scores[j]; }
  for (var k = 0u; k < hd; k++) {
    var acc = 0.0;
    for (var j = 0u; j < d.a; j++) { acc += (scores[j] / sum) * F3[cb + j * d.c + off + k]; }
    O1[qb + k] = acc;
  }
}

// addv_s / gelu_s: a=总元素数（batch 已摊平）
@compute @workgroup_size(64) fn addv_s(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= d.a) { return; }
  O1[g.x] = F1[g.x] + F2[g.x];
}
@compute @workgroup_size(64) fn gelu_s(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= d.a) { return; }
  let x = F1[g.x];
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x*x*x), -9.0, 9.0);
  O1[g.x] = 0.5 * x * (1.0 + tanh(t));
}

// sample: GPU 端 top-k 采样，结果直接写回 ids —— 每 workgroup 一个 batch 项
//   a=V b=topK c=pos d2=mode(0=温度 1=前两字均匀top4) | F1=logits F2=随机表 OU=ids
//   64 线程分段各求局部 top-K 再归并（单线程扫 6379 会成为新瓶颈）
var<workgroup> tv: array<f32, 512>;
var<workgroup> ti: array<u32, 512>;
@compute @workgroup_size(64) fn sample(@builtin(workgroup_id) wid: vec3<u32>,
                                       @builtin(local_invocation_id) lid: vec3<u32>) {
  let bi = wid.x;
  let lb = bi * d.a;
  let K = d.b;
  var vals: array<f32, 8>;
  var idxs: array<u32, 8>;
  for (var i = 0u; i < K; i++) { vals[i] = -1e30; idxs[i] = 0u; }
  for (var j = lid.x; j < d.a; j += 64u) {
    let v = F1[lb + j];
    if (v > vals[K - 1u]) {
      var p = K - 1u;
      while (p > 0u && v > vals[p - 1u]) { vals[p] = vals[p - 1u]; idxs[p] = idxs[p - 1u]; p = p - 1u; }
      vals[p] = v; idxs[p] = j;
    }
  }
  for (var i = 0u; i < K; i++) { tv[lid.x * 8u + i] = vals[i]; ti[lid.x * 8u + i] = idxs[i]; }
  workgroupBarrier();
  if (lid.x != 0u) { return; }

  var fv: array<f32, 8>;
  var fi: array<u32, 8>;
  for (var i = 0u; i < K; i++) { fv[i] = -1e30; fi[i] = 0u; }
  for (var t = 0u; t < 64u; t++) {
    for (var i = 0u; i < K; i++) {
      let v = tv[t * 8u + i];
      if (v > fv[K - 1u]) {
        let id = ti[t * 8u + i];
        var p = K - 1u;
        while (p > 0u && v > fv[p - 1u]) { fv[p] = fv[p - 1u]; fi[p] = fi[p - 1u]; p = p - 1u; }
        fv[p] = v; fi[p] = id;
      }
    }
  }
  let rb = bi * 80u;                 // 每 batch 项独立随机数表
  let r = F2[rb + d.c];
  var chosen = fi[0];
  if (d.d2 == 1u) {
    let n = min(4u, K);
    chosen = fi[min(u32(r * f32(n)), n - 1u)];
  } else {
    let invT = F2[rb + 72u];
    let mx = fv[0] * invT;
    var ps: array<f32, 8>;
    var sum = 0.0;
    for (var i = 0u; i < K; i++) { ps[i] = exp(fv[i] * invT - mx); sum += ps[i]; }
    var acc = 0.0;
    chosen = fi[K - 1u];
    for (var i = 0u; i < K; i++) {
      acc += ps[i] / sum;
      if (r <= acc) { chosen = fi[i]; break; }
    }
  }
  OU[bi * 72u + d.c + 1u] = chosen;  // ids 步长固定 72（≥T，对齐用）
}
`;

function softmax(arr) {
  const mx = Math.max(...arr);
  const ex = arr.map((x) => Math.exp(x - mx));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((e) => e / s);
}

// chars 可省：新版权重把字表存在 meta.vocab 里，优先用它。
// 以前由调用方从语料现算字表，一旦语料换了（加宋诗）就与旧权重错位，
// 输出变乱码且不报错——字表必须随模型走。
export async function createPoet(meta, binBuffer, chars) {
  const cfg = meta.cfg;
  chars = meta.vocab || chars;
  if (!chars) throw new Error("缺字表：meta 里没有 vocab，调用时也未传入 chars");
  if (chars.length !== cfg.vocabSize) {
    throw new Error(`字表不匹配：权重期望 ${cfg.vocabSize} 字，实际拿到 ${chars.length} 字。` +
      `语料可能换过了，用匹配该权重的语料，或重新训练。`);
  }
  const V = cfg.vocabSize, E = cfg.nEmbd, H = cfg.nHead, L = cfg.nLayer;
  const BS = cfg.blockSize, T = BS - 1, FF = 4 * E;
  const stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
  const NL = stoi["\n"];

  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU 不可用");
  const device = await adapter.requestDevice({
    requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
  });

  const pipelines = {};
  for (const [src, names] of [
    [MM_WGSL, ["mm_nn", "mm_nt"]],
    [OPS_WGSL, ["embed_fwd", "ln_fwd", "attn_fwd", "addv", "gelu_fwd"]],
    [STEP_WGSL, ["embed_step", "ln_s", "mv", "mvT", "attn_step", "addv_s", "gelu_s", "sample"]],
  ]) {
    const mod = device.createShaderModule({ code: src });
    for (const n of names) pipelines[n] = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: n } });
  }

  const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const buf = (n) => device.createBuffer({ size: Math.max(n * 4, 16), usage: SU });

  // ---- 权重上显存（ln 的 g/b 拼成一个缓冲，方便单绑定） ----
  const W = {};   // name → GPUBuffer
  {
    const raw = new Float32Array(binBuffer);
    let off = 0;
    const slices = {};
    for (const t of meta.tensors) { slices[t.name] = raw.subarray(off, off + t.rows * t.cols); off += t.rows * t.cols; }
    const up = (name, arr) => { const b = buf(arr.length); device.queue.writeBuffer(b, 0, arr.slice()); W[name] = b; };
    up("wte", slices["wte"]); up("wpe", slices["wpe"]);
    const cat = (a, b) => { const r = new Float32Array(a.length + b.length); r.set(a); r.set(b, a.length); return r; };
    up("lnF", cat(slices["lnFg"], slices["lnFb"]));
    for (let l = 0; l < L; l++) {
      up(`b${l}.ln1`, cat(slices[`b${l}.ln1g`], slices[`b${l}.ln1b`]));
      up(`b${l}.ln2`, cat(slices[`b${l}.ln2g`], slices[`b${l}.ln2b`]));
      for (const w of ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]) up(`b${l}.${w}`, slices[`b${l}.${w}`]);
    }
  }

  // ---- 激活缓冲（单序列 T 行，双缓冲残差流 X→Xa→Xb→Xa→Xb...）----
  const act = {
    ids: device.createBuffer({ size: T * 4, usage: SU }),
    X: buf(T * E), Xa: buf(T * E), Xb: buf(T * E), h: buf(T * E),
    q: buf(T * E), k: buf(T * E), v: buf(T * E), qkv: buf(3 * T * E),
    ao: buf(T * E), proj: buf(T * E), m1: buf(T * FF), g1: buf(T * FF), m2: buf(T * E),
    hF: buf(T * E), hRow: buf(E), logits: buf(V),
    read: device.createBuffer({ size: V * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
  };

  const uCache = new Map(), bgCache = new Map();
  let bufSeq = 0; const bufIds = new WeakMap();
  const bid = (b) => { if (!bufIds.has(b)) bufIds.set(b, ++bufSeq); return bufIds.get(b); };
  const uniform = (key, arr) => {
    if (!uCache.has(key)) {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Uint32Array([...arr, 0, 0, 0].slice(0, 4)));
      uCache.set(key, b);
    }
    return uCache.get(key);
  };
  const dummy = buf(4);
  function run(pass, kind, dims, entries, threads, wg) {
    const u = uniform(kind + "|" + dims.join(","), dims);
    const full = [u, ...entries];
    const mask = BIND_MASK[kind];
    const key = kind + "|" + mask.map((i) => bid(full[i])).join(",");
    if (!bgCache.has(key)) {
      bgCache.set(key, device.createBindGroup({
        layout: pipelines[kind].getBindGroupLayout(0),
        entries: mask.map((i) => ({ binding: i, resource: { buffer: full[i] } })),
      }));
    }
    pass.setPipeline(pipelines[kind]);
    pass.setBindGroup(0, bgCache.get(key));
    pass.dispatchWorkgroups(...(kind.startsWith("mm") ? threads : [Math.ceil(threads / wg)]));
  }
  const mm = (pass, kind, A, B, C, n, k, m) =>
    run(pass, kind, [n, k, m, 0], [A, B, C], [Math.ceil(m / 64), Math.ceil(n / 64)]);

  // ---- KV Cache + 批处理缓冲（IDS 步长固定 72，与 sample 内核约定一致）----
  const MAXB = 8, IDS = 72;
  const kv = [];
  for (let l = 0; l < L; l++) kv.push({ k: buf(MAXB * T * E), v: buf(MAXB * T * E) });
  const st = {
    ids: device.createBuffer({ size: MAXB * IDS * 4, usage: SU }),
    res0: buf(MAXB * E), res1: buf(MAXB * E), xa: buf(MAXB * E), h: buf(MAXB * E),
    q: buf(MAXB * E), ao: buf(MAXB * E), proj: buf(MAXB * E),
    m1: buf(MAXB * FF), g1: buf(MAXB * FF), m2: buf(MAXB * E),
    hF: buf(MAXB * E), logits: buf(MAXB * V),
    rnd: buf(MAXB * 80),
    read: device.createBuffer({ size: MAXB * V * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    readIds: device.createBuffer({ size: MAXB * IDS * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
  };

  // 编码位置 pos 的一步前向（batch 内所有序列同时推进）
  function encodeToken(pass, pos, nb) {
    run(pass, "embed_step", [E, IDS, pos], [st.ids, W.wte, W.wpe, st.res0], nb * E, 64);
    let Xin = st.res0;
    for (let l = 0; l < L; l++) {
      const out = l % 2 === 0 ? st.res1 : st.res0;   // 残差双缓冲交替
      run(pass, "ln_s", [E], [dummy, Xin, W[`b${l}.ln1`], st.h], nb, 1);
      run(pass, "mv", [E, E, 0, 1], [dummy, st.h, W[`b${l}.Wq`], st.q], nb * E, 64);
      run(pass, "mv", [E, E, pos, T], [dummy, st.h, W[`b${l}.Wk`], kv[l].k], nb * E, 64);
      run(pass, "mv", [E, E, pos, T], [dummy, st.h, W[`b${l}.Wv`], kv[l].v], nb * E, 64);
      run(pass, "attn_step", [pos + 1, H, E, T], [dummy, st.q, kv[l].k, st.ao, kv[l].v], nb * H, 64);
      run(pass, "mv", [E, E, 0, 1], [dummy, st.ao, W[`b${l}.Wo`], st.proj], nb * E, 64);
      run(pass, "addv_s", [nb * E], [dummy, Xin, st.proj, st.xa], nb * E, 64);
      run(pass, "ln_s", [E], [dummy, st.xa, W[`b${l}.ln2`], st.h], nb, 1);
      run(pass, "mv", [E, FF, 0, 1], [dummy, st.h, W[`b${l}.W1`], st.m1], nb * FF, 64);
      run(pass, "gelu_s", [nb * FF], [dummy, st.m1, dummy, st.g1], nb * FF, 64);
      run(pass, "mv", [FF, E, 0, 1], [dummy, st.g1, W[`b${l}.W2`], st.m2], nb * E, 64);
      run(pass, "addv_s", [nb * E], [dummy, st.xa, st.m2, out], nb * E, 64);
      Xin = out;
    }
    run(pass, "ln_s", [E], [dummy, Xin, W.lnF, st.hF], nb, 1);
    run(pass, "mvT", [E, V], [dummy, st.hF, W.wte, st.logits], nb * V, 64);
  }

  // KV Cache 前向：喂入 [from,to] 区间，返回 batch 项 0 的末位 logits（对拍用）
  async function stepRange(from, to, nb = 1) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let pos = from; pos <= to; pos++) encodeToken(pass, pos, nb);
    pass.end();
    enc.copyBufferToBuffer(st.logits, 0, st.read, 0, V * 4);
    device.queue.submit([enc.finish()]);
    await st.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.read.getMappedRange().slice(0, V * 4));
    st.read.unmap();
    return out;
  }

  // ---- 前向：返回位置 len-1 的 logits ----
  async function logitsAt(ids) {
    const len = ids.length;
    const padded = new Uint32Array(T);
    padded.set(ids.slice(0, T));
    device.queue.writeBuffer(act.ids, 0, padded);
    const enc = device.createCommandEncoder();
    {
      const p = enc.beginComputePass();
      run(p, "embed_fwd", [T, BS, E], [act.ids, W.wte, W.wpe, act.X], T * E, 256);
      p.end();
    }
    let Xin = act.X;
    for (let l = 0; l < L; l++) {
      // passA: ln1 + 三个投影
      const pA = enc.beginComputePass();
      run(pA, "ln_fwd", [T, E], [dummy, Xin, W[`b${l}.ln1`], act.h], T, 64);
      mm(pA, "mm_nn", act.h, W[`b${l}.Wq`], act.q, T, E, E);
      mm(pA, "mm_nn", act.h, W[`b${l}.Wk`], act.k, T, E, E);
      mm(pA, "mm_nn", act.h, W[`b${l}.Wv`], act.v, T, E, E);
      pA.end();
      // 拼接 q/k/v 到一个缓冲（attn 内核单绑定读三段）
      enc.copyBufferToBuffer(act.q, 0, act.qkv, 0, T * E * 4);
      enc.copyBufferToBuffer(act.k, 0, act.qkv, T * E * 4, T * E * 4);
      enc.copyBufferToBuffer(act.v, 0, act.qkv, 2 * T * E * 4, T * E * 4);
      // passB: attn + 残差 + MLP + 残差
      const pB = enc.beginComputePass();
      run(pB, "attn_fwd", [T, H, E], [dummy, act.qkv, dummy, act.ao], H * T, 64);
      mm(pB, "mm_nn", act.ao, W[`b${l}.Wo`], act.proj, T, E, E);
      run(pB, "addv", [T * E], [dummy, Xin, act.proj, act.Xa], T * E, 256);
      run(pB, "ln_fwd", [T, E], [dummy, act.Xa, W[`b${l}.ln2`], act.h], T, 64);
      mm(pB, "mm_nn", act.h, W[`b${l}.W1`], act.m1, T, E, FF);
      run(pB, "gelu_fwd", [T * FF], [dummy, act.m1, dummy, act.g1], T * FF, 256);
      mm(pB, "mm_nn", act.g1, W[`b${l}.W2`], act.m2, T, FF, E);
      run(pB, "addv", [T * E], [dummy, act.Xa, act.m2, act.Xb], T * E, 256);
      pB.end();
      Xin = act.Xb;
    }
    {
      const p = enc.beginComputePass();
      run(p, "ln_fwd", [T, E], [dummy, Xin, W.lnF, act.hF], T, 64);
      p.end();
    }
    // 只算最后有效行的 logits：拷出该行 → 1×V 矩阵乘
    enc.copyBufferToBuffer(act.hF, (len - 1) * E * 4, act.hRow, 0, E * 4);
    {
      const p = enc.beginComputePass();
      mm(p, "mm_nt", act.hRow, W.wte, act.logits, 1, E, V);
      p.end();
    }
    enc.copyBufferToBuffer(act.logits, 0, act.read, 0, V * 4);
    device.queue.submit([enc.finish()]);
    await act.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(act.read.getMappedRange().slice(0));
    act.read.unmap();
    return out;
  }

  // ---- 采样（KV Cache 路径；策略与 CPU 版 generate 完全一致）----
  const encodeS = (s) => [...s].map((c) => stoi[c]);
  const decodeS = (ids) => ids.map((i) => chars[i]).join("");

  function pick(logits, { explore, s, temperature, topK }) {
    const last = Array.from(logits);
    if (explore && s < 2) {
      const idx = last.map((v, i) => i).sort((a, b) => last[b] - last[a]).slice(0, 4);
      return idx[Math.floor(Math.random() * idx.length)];
    }
    const idx = last.map((v, i) => i).sort((a, b) => last[b] - last[a]).slice(0, topK);
    const probs = softmax(idx.map((i) => last[i] / temperature));
    let r = Math.random(), next = idx[0];
    for (let i = 0; i < idx.length; i++) { r -= probs[i]; if (r <= 0) { next = idx[i]; break; } }
    return next;
  }

  // 批量生成：count 首诗同时推进。派发指令数与 count 无关，
  // 而耗时几乎全在派发开销上 —— 所以出 4 首 ≈ 出 1 首的时间
  const CHUNK = 12;
  async function generateBatch(start, count = 1, { temperature = 0.6, topK = 5, maxNew = 0 } = {}) {
    const nb = Math.min(count, MAXB);
    const ids0 = encodeS("\n" + start);
    const n0 = ids0.length;
    const explore = start.length < 2;
    const limit = Math.min(T - 1, n0 + (maxNew || 65 - start.length));

    // 每序列独立随机数表（同一 start 也能生成不同的诗）
    const rnd = new Float32Array(MAXB * 80);
    for (let b = 0; b < nb; b++) {
      for (let i = 0; i < 73; i++) rnd[b * 80 + i] = Math.random();
      rnd[b * 80 + 72] = 1 / temperature;
    }
    device.queue.writeBuffer(st.rnd, 0, rnd);
    const padded = new Uint32Array(MAXB * IDS);
    for (let b = 0; b < nb; b++) padded.set(ids0, b * IDS);
    device.queue.writeBuffer(st.ids, 0, padded);

    let cur = padded;
    let reached = 0;
    for (let from = 0; from < limit; from += CHUNK) {
      const to = Math.min(from + CHUNK, limit);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let pos = from; pos < to; pos++) {
        encodeToken(pass, pos, nb);
        if (pos >= n0 - 1) {
          const s = pos - (n0 - 1);
          run(pass, "sample", [V, topK, pos, explore && s < 2 ? 1 : 0],
            [dummy, st.logits, st.rnd, dummy, dummy, st.ids], nb, 1);
        }
      }
      pass.end();
      enc.copyBufferToBuffer(st.ids, 0, st.readIds, 0, MAXB * IDS * 4);
      device.queue.submit([enc.finish()]);
      await st.readIds.mapAsync(GPUMapMode.READ);
      cur = new Uint32Array(st.readIds.getMappedRange().slice(0));
      st.readIds.unmap();
      reached = to;
      // 所有序列都已收尾则提前结束
      let allDone = true;
      for (let b = 0; b < nb && allDone; b++) {
        let hit = false;
        for (let i = n0; i <= to; i++) if (cur[b * IDS + i] === NL) { hit = true; break; }
        allDone = hit;
      }
      if (allDone) break;
    }

    const out = [];
    for (let b = 0; b < nb; b++) {
      const seq = [];
      for (let i = 0; i <= reached && i < IDS; i++) {
        const t = cur[b * IDS + i];
        if (i >= n0 && t === NL) break;
        seq.push(t);
      }
      out.push(decodeS(seq).trim());
    }
    return out;
  }

  const generate = async (start, opts) => (await generateBatch(start, 1, opts))[0];

  // 逐 token 往返版（保留：便于对照延迟开销）
  async function generateStepwise(start, { temperature = 0.6, topK = 5, maxNew = 0 } = {}) {
    const ids = encodeS("\n" + start);
    const explore = start.length < 2;
    const maxNewTokens = maxNew || 65 - start.length;
    device.queue.writeBuffer(st.ids, 0, new Uint32Array(ids));
    let logits = await stepRange(0, ids.length - 1);
    for (let s = 0; s < maxNewTokens; s++) {
      const next = pick(logits, { explore, s, temperature, topK });
      if (next === NL || ids.length >= T) break;
      const pos = ids.length;
      ids.push(next);
      device.queue.writeBuffer(st.ids, pos * 4, new Uint32Array([next]));
      logits = await stepRange(pos, pos);
    }
    return decodeS(ids).trim();
  }

  // 旧路径（每步重算全序列）——留作数值对拍与性能对比
  async function generateFull(start, { temperature = 0.6, topK = 5, maxNew = 0 } = {}) {
    let ids = encodeS("\n" + start);
    const explore = start.length < 2;
    const maxNewTokens = maxNew || 65 - start.length;
    for (let s = 0; s < maxNewTokens; s++) {
      if (ids.length >= T) break;
      const next = pick(await logitsAt(ids), { explore, s, temperature, topK });
      if (next === NL) break;
      ids.push(next);
    }
    return decodeS(ids).trim();
  }

  // KV Cache 单次前向（对拍用）：喂入整段 ids，返回末位 logits
  async function logitsKV(ids) {
    device.queue.writeBuffer(st.ids, 0, new Uint32Array(ids));
    return await stepRange(0, ids.length - 1);
  }

  // 基准探针：单 token 增量（cache 须已填好）/ 一个 pass 内串 n 个 token
  const stepOne = (pos) => stepRange(pos, pos);
  const stepChunk = (from, to) => stepRange(from, to);

  return { generate, generateBatch, generateStepwise, generateFull, logitsAt, logitsKV, stepOne, stepChunk, device, cfg };
}
