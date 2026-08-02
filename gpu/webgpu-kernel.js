// ============================================================
// webgpu-kernel.js —— WebGPU 矩阵乘内核（Deno / 浏览器通用，零依赖）
//
// 终极形态·阶段1：WGSL compute shader 实现 matmul，
// 带 CPU 数值对齐自检 + 分档基准测试。
//
// 运行: deno run --allow-none webgpu-kernel.js
//
// 设计要点:
//   - 16×16 tile 分块：每个 workgroup 算 C 的一个 16×16 子块，
//     A/B 子块先搬进 workgroup 共享内存再复用（省显存带宽）
//   - 数据常驻显存测试：真实训练中权重不回 CPU，
//     基准分"每次搬数据"和"数据常驻"两种口径，诚实反映差距
// ============================================================

const TILE = 16;

// V2 内核：寄存器分块 —— 每线程算 4×4 输出，workgroup 16×16 线程共算 64×64 子块
const WGSL_V2 = /* wgsl */ `
struct Dims { n: u32, k: u32, m: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// 64×16 的 A 子块 + 16×64 的 B 子块
var<workgroup> tileA: array<f32, 1024>;
var<workgroup> tileB: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowBase = wid.y * 64u + lid.y * 4u;   // 本线程负责的 4 行起点
  let colBase = wid.x * 64u + lid.x * 4u;   // 本线程负责的 4 列起点
  var acc: array<f32, 16>;                  // 4×4 寄存器累加器
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  let numTiles = (dims.k + 15u) / 16u;
  let tid = lid.y * 16u + lid.x;            // 0..255

  for (var t = 0u; t < numTiles; t = t + 1u) {
    // 协作装载：256 线程各搬 4 个元素，装满 64×16 的 A 块和 16×64 的 B 块
    for (var s = 0u; s < 4u; s = s + 1u) {
      let idx = tid * 4u + s;               // 0..1023
      let ar = idx / 16u;  let ac = idx % 16u;       // A: 64行×16列
      let aRow = wid.y * 64u + ar;  let aCol = t * 16u + ac;
      if (aRow < dims.n && aCol < dims.k) { tileA[idx] = A[aRow * dims.k + aCol]; }
      else { tileA[idx] = 0.0; }
      let br = idx / 64u;  let bc = idx % 64u;       // B: 16行×64列
      let bRow = t * 16u + br;  let bCol = wid.x * 64u + bc;
      if (bRow < dims.k && bCol < dims.m) { tileB[idx] = B[bRow * dims.m + bCol]; }
      else { tileB[idx] = 0.0; }
    }
    workgroupBarrier();

    // 4×4 寄存器块累加：每个 k 取 A 的 4 个、B 的 4 个，做 16 次乘加
    for (var i = 0u; i < 16u; i = i + 1u) {
      let a0 = tileA[(lid.y * 4u + 0u) * 16u + i];
      let a1 = tileA[(lid.y * 4u + 1u) * 16u + i];
      let a2 = tileA[(lid.y * 4u + 2u) * 16u + i];
      let a3 = tileA[(lid.y * 4u + 3u) * 16u + i];
      let b0 = tileB[i * 64u + lid.x * 4u + 0u];
      let b1 = tileB[i * 64u + lid.x * 4u + 1u];
      let b2 = tileB[i * 64u + lid.x * 4u + 2u];
      let b3 = tileB[i * 64u + lid.x * 4u + 3u];
      acc[0]  += a0 * b0;  acc[1]  += a0 * b1;  acc[2]  += a0 * b2;  acc[3]  += a0 * b3;
      acc[4]  += a1 * b0;  acc[5]  += a1 * b1;  acc[6]  += a1 * b2;  acc[7]  += a1 * b3;
      acc[8]  += a2 * b0;  acc[9]  += a2 * b1;  acc[10] += a2 * b2;  acc[11] += a2 * b3;
      acc[12] += a3 * b0;  acc[13] += a3 * b1;  acc[14] += a3 * b2;  acc[15] += a3 * b3;
    }
    workgroupBarrier();
  }

  // 写回 4×4 结果
  for (var r = 0u; r < 4u; r = r + 1u) {
    let row = rowBase + r;
    if (row >= dims.n) { continue; }
    for (var c = 0u; c < 4u; c = c + 1u) {
      let col = colBase + c;
      if (col < dims.m) { C[row * dims.m + col] = acc[r * 4u + c]; }
    }
  }
}
`;

const WGSL = /* wgsl */ `
struct Dims { n: u32, k: u32, m: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

var<workgroup> tileA: array<f32, ${TILE * TILE}>;
var<workgroup> tileB: array<f32, ${TILE * TILE}>;

@compute @workgroup_size(${TILE}, ${TILE})
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = gid.y;            // C 的行
  let col = gid.x;            // C 的列
  var acc = 0.0;

  let numTiles = (dims.k + ${TILE}u - 1u) / ${TILE}u;
  for (var t = 0u; t < numTiles; t = t + 1u) {
    // 协作装载 A、B 的 16×16 子块到共享内存
    let aCol = t * ${TILE}u + lid.x;
    let bRow = t * ${TILE}u + lid.y;
    if (row < dims.n && aCol < dims.k) {
      tileA[lid.y * ${TILE}u + lid.x] = A[row * dims.k + aCol];
    } else {
      tileA[lid.y * ${TILE}u + lid.x] = 0.0;
    }
    if (bRow < dims.k && col < dims.m) {
      tileB[lid.y * ${TILE}u + lid.x] = B[bRow * dims.m + col];
    } else {
      tileB[lid.y * ${TILE}u + lid.x] = 0.0;
    }
    workgroupBarrier();

    for (var i = 0u; i < ${TILE}u; i = i + 1u) {
      acc = acc + tileA[lid.y * ${TILE}u + i] * tileB[i * ${TILE}u + lid.x];
    }
    workgroupBarrier();
  }

  if (row < dims.n && col < dims.m) {
    C[row * dims.m + col] = acc;
  }
}
`;

// ---------- GPU 上下文 ----------

export async function initGPU(version = 2) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU 不可用");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  const module = device.createShaderModule({ code: version === 2 ? WGSL_V2 : WGSL });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "matmul" },
  });
  return { device, pipeline, version };
}

// ---------- 显存缓冲工具 ----------

export function gpuBuffer(device, sizeBytes, usage) {
  return device.createBuffer({ size: Math.max(sizeBytes, 16), usage });
}

export function uploadMatrix(device, buf, flatData) {
  device.queue.writeBuffer(buf, 0, flatData.buffer, flatData.byteOffset, flatData.byteLength);
}

// 一次 matmul 派发（数据已在显存，只编排命令）
export function encodeMatmul(device, pipeline, pass, dimsBuf, aBuf, bBuf, cBuf, n, k, m, version = 2) {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dimsBuf } },
      { binding: 1, resource: { buffer: aBuf } },
      { binding: 2, resource: { buffer: bBuf } },
      { binding: 3, resource: { buffer: cBuf } },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const tile = version === 2 ? 64 : TILE;   // V2: 每 workgroup 算 64×64
  pass.dispatchWorkgroups(Math.ceil(m / tile), Math.ceil(n / tile));
}

// 便捷版：CPU 数据进出（含搬运开销，用于自检和"搬运口径"基准）
export async function matmulOnce(ctx, Aflat, Bflat, n, k, m) {
  const { device, pipeline } = ctx;
  const dimsBuf = gpuBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(dimsBuf, 0, new Uint32Array([n, k, m, 0]));
  const aBuf = gpuBuffer(device, Aflat.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const bBuf = gpuBuffer(device, Bflat.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const cBuf = gpuBuffer(device, n * m * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readBuf = gpuBuffer(device, n * m * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  uploadMatrix(device, aBuf, Aflat);
  uploadMatrix(device, bBuf, Bflat);

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  encodeMatmul(device, pipeline, pass, dimsBuf, aBuf, bBuf, cBuf, n, k, m, ctx.version);
  pass.end();
  enc.copyBufferToBuffer(cBuf, 0, readBuf, 0, n * m * 4);
  device.queue.submit([enc.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  for (const b of [dimsBuf, aBuf, bBuf, cBuf, readBuf]) b.destroy();
  return out;
}

// ---------- CPU 参照实现 ----------

function cpuMatmul(Aflat, Bflat, n, k, m) {
  const C = new Float32Array(n * m);
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      const a = Aflat[i * k + p];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i * m + j] += a * Bflat[p * m + j];
    }
  return C;
}

const rand = (len) => Float32Array.from({ length: len }, () => Math.random() * 2 - 1);

// ---------- 自检 + 基准 ----------

async function main() {
  console.log("== WebGPU 内核 阶段1 (V2: 4×4 寄存器分块) ==\n");
  const ctx = await initGPU(2);
  console.log("GPU device 就绪\n");

  // 1) 数值自检：4 种 Transformer 实际形状
  console.log("― 数值自检（vs CPU 逐元素） ―");
  let allPass = true;
  for (const [n, k, m] of [[26, 192, 768], [26, 768, 192], [26, 384, 1536], [832, 384, 384]]) {
    const A = rand(n * k), B = rand(k * m);
    const gpu = await matmulOnce(ctx, A, B, n, k, m);
    const cpu = cpuMatmul(A, B, n, k, m);
    let maxDiff = 0;
    for (let i = 0; i < gpu.length; i++) maxDiff = Math.max(maxDiff, Math.abs(gpu[i] - cpu[i]));
    const ok = maxDiff < 1e-3;
    allPass = allPass && ok;
    console.log(`  ${n}×${k}·${k}×${m}: 最大误差 ${maxDiff.toExponential(2)} ${ok ? "✅" : "❌"}`);
  }
  if (!allPass) { console.error("自检失败，停止"); Deno.exit(1); }

  // 2) 基准 A：小矩阵 + 每次搬数据（= 当前 CPU 模型的形状，诚实口径）
  console.log("\n― 基准A: 小矩阵·含CPU↔GPU搬运（当前模型形状 26×384·384×1536） ―");
  {
    const n = 26, k = 384, m = 1536;
    const A = rand(n * k), B = rand(k * m);
    const N = 50;
    const t0 = performance.now();
    for (let i = 0; i < N; i++) await matmulOnce(ctx, A, B, n, k, m);
    const ms = (performance.now() - t0) / N;
    const gflops = (2 * n * k * m) / (ms / 1000) / 1e9;
    console.log(`  ${ms.toFixed(2)}ms/次  →  ${gflops.toFixed(1)} GFLOPS`);
  }

  // 3) 基准 B：大 batch + 数据常驻显存（= 终极形态训练的真实口径）
  console.log("\n― 基准B: 大batch·数据常驻显存（832×512·512×2048，连续派发） ―");
  {
    const n = 832, k = 512, m = 2048;   // batch 32 首诗 × 26 token
    const { device, pipeline } = ctx;
    const dimsBuf = gpuBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(dimsBuf, 0, new Uint32Array([n, k, m, 0]));
    const aBuf = gpuBuffer(device, n * k * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bBuf = gpuBuffer(device, k * m * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const cBuf = gpuBuffer(device, n * m * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    uploadMatrix(device, aBuf, rand(n * k));
    uploadMatrix(device, bBuf, rand(k * m));

    // 预热
    for (let w = 0; w < 3; w++) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      encodeMatmul(device, pipeline, pass, dimsBuf, aBuf, bBuf, cBuf, n, k, m, ctx.version);
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    const N = 200;
    const t0 = performance.now();
    // 一次 encoder 打包 20 个派发再 submit，摄薄指挥开销
    for (let i = 0; i < N / 20; i++) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let j = 0; j < 20; j++)
        encodeMatmul(device, pipeline, pass, dimsBuf, aBuf, bBuf, cBuf, n, k, m, ctx.version);
      pass.end();
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const ms = (performance.now() - t0) / N;
    const gflops = (2 * n * k * m) / (ms / 1000) / 1e9;
    console.log(`  ${ms.toFixed(2)}ms/次  →  ${gflops.toFixed(1)} GFLOPS`);
    console.log(`\n  参照: WASM+Worker 实测 ~40 GFLOPS（CPU 全家桶）`);
    console.log(`  达标线(20×): 800 GFLOPS`);
  }
}

if (import.meta.main) await main();
