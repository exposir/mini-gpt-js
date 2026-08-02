// ============================================================
// wasm-kernel.js —— 手写 WebAssembly SIMD 矩阵乘法核（零依赖）
//
// 不用 wat2wasm/AssemblyScript 等任何工具链：在 JS 里用迷你汇编器
// 直接吐出 wasm 二进制字节，Node 内置 WebAssembly 即可编译。
//
// 核心加速：f32x4 SIMD —— 一条指令同时算 4 个浮点乘加，
//           替代 JS 的逐元素三重 for 循环。
//
// 对外接口:
//   matmul(A, B)   A(n×k)·B(k×m) → C(n×m)，A/B/C 均为“行数组 of Float32Array”
//                  要求 m 为 4 的倍数（本项目 nEmbd=192、4*nEmbd=768 均满足）
//   selfTest()     与纯 JS 实现逐元素比对，返回是否数值一致
//   USE_WASM       编译或自检失败时为 false，调用方应回退 JS
// ============================================================

// ---------- LEB128 编码（wasm 的变长整数格式）----------
function uleb(n) {
  const b = [];
  do { let x = n & 0x7f; n >>>= 7; if (n) x |= 0x80; b.push(x); } while (n);
  return b;
}
function sleb(n) {
  const b = []; let more = 1;
  while (more) {
    let x = n & 0x7f; n >>= 7;
    if ((n === 0 && !(x & 0x40)) || (n === -1 && (x & 0x40))) more = 0; else x |= 0x80;
    b.push(x);
  }
  return b;
}

// ---------- 迷你指令汇编器（只覆盖本核用到的 opcode）----------
const I = {
  get: i => [0x20, ...uleb(i)],           // local.get
  set: i => [0x21, ...uleb(i)],           // local.set
  i32c: v => [0x41, ...sleb(v)],          // i32.const
  iadd: [0x6a], imul: [0x6c], ge_s: [0x4e],
  f32load: [0x2a, 2, 0],                  // f32.load  (align=4B, offset=0)
  v128load: [0xfd, 0x00, 2, 0],           // v128.load
  v128store: [0xfd, 0x0b, 2, 0],          // v128.store
  splat: [0xfd, 0x13],                    // f32x4.splat
  fmul: [0xfd, 0xe6, 0x01],               // f32x4.mul  (subop 230)
  fadd: [0xfd, 0xe4, 0x01],               // f32x4.add  (subop 228)
  block: [0x02, 0x40], loop: [0x03, 0x40], end: [0x0b],
  br: d => [0x0c, ...uleb(d)], br_if: d => [0x0d, ...uleb(d)],
};

// 局部变量索引（0-5 为参数）
const aPtr = 0, bPtr = 1, cPtr = 2, N = 3, K = 4, M = 5;
const i_ = 6, p_ = 7, j_ = 8, aRow = 9, bRow = 10, cRow = 11, cAddr = 12, bAddr = 13, av = 14;

// C[i][j:j+4] += A[i][p] * B[p][j:j+4]，三重循环，最内层 SIMD 一次算 4 列
function buildModule() {
  const body = [].concat(
    I.i32c(0), I.set(i_),
    I.block, I.loop,                                            // ── 行循环 i ──
      I.get(i_), I.get(N), I.ge_s, I.br_if(1),                  // if i>=n break
      // cRow = cPtr + i*m*4 ; aRow = aPtr + i*k*4
      I.get(cPtr), I.get(i_), I.get(M), I.imul, I.i32c(4), I.imul, I.iadd, I.set(cRow),
      I.get(aPtr), I.get(i_), I.get(K), I.imul, I.i32c(4), I.imul, I.iadd, I.set(aRow),
      I.i32c(0), I.set(p_),
      I.block, I.loop,                                          // ── 中层循环 p ──
        I.get(p_), I.get(K), I.ge_s, I.br_if(1),                // if p>=k break
        // av = splat( A[aRow + p*4] )
        I.get(aRow), I.get(p_), I.i32c(4), I.imul, I.iadd, I.f32load, I.splat, I.set(av),
        // bRow = bPtr + p*m*4
        I.get(bPtr), I.get(p_), I.get(M), I.imul, I.i32c(4), I.imul, I.iadd, I.set(bRow),
        I.i32c(0), I.set(j_),
        I.block, I.loop,                                        // ── 内层循环 j（SIMD）──
          I.get(j_), I.get(M), I.ge_s, I.br_if(1),              // if j>=m break
          I.get(cRow), I.get(j_), I.i32c(4), I.imul, I.iadd, I.set(cAddr),
          I.get(bRow), I.get(j_), I.i32c(4), I.imul, I.iadd, I.set(bAddr),
          // C[cAddr] = C[cAddr] + av * B[bAddr]   （4 路并行）
          I.get(cAddr),
          I.get(cAddr), I.v128load,
          I.get(av),
          I.get(bAddr), I.v128load,
          I.fmul, I.fadd, I.v128store,
          I.get(j_), I.i32c(4), I.iadd, I.set(j_), I.br(0),     // j+=4; continue
        I.end, I.end,
        I.get(p_), I.i32c(1), I.iadd, I.set(p_), I.br(0),       // p+=1; continue
      I.end, I.end,
      I.get(i_), I.i32c(1), I.iadd, I.set(i_), I.br(0),         // i+=1; continue
    I.end, I.end,
    I.end,                                                      // func end
  );

  const locals = [0x02, 0x08, 0x7f, 0x01, 0x7b];   // 8×i32 + 1×v128
  const funcBody = [...locals, ...body];
  const codeEntry = [...uleb(funcBody.length), ...funcBody];

  const sec = (id, content) => [id, ...uleb(content.length), ...content];
  const bytes = [
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,             // magic + version
    ...sec(1, [0x01, 0x60, 0x06, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x00]),  // type: (i32×6)->()
    ...sec(3, [0x01, 0x00]),                                    // function: func0 uses type0
    ...sec(5, [0x01, 0x00, 0x80, 0x02]),                       // memory: min 256 页 = 16MB
    ...sec(7, [0x02,                                            // exports:
      0x03, 0x6d, 0x65, 0x6d, 0x02, 0x00,                       //   "mem" → memory 0
      0x06, 0x6d, 0x61, 0x74, 0x6d, 0x75, 0x6c, 0x00, 0x00]),   //   "matmul" → func 0
    ...sec(10, [0x01, ...codeEntry]),                           // code
  ];
  return new Uint8Array(bytes);
}

// ---------- 实例化 + 内存分区 ----------
const A_OFF = 0, B_OFF = 0x400000, C_OFF = 0xA00000;   // A:4MB | B:6MB | C:6MB（字节偏移）
let USE_WASM = false, mm = null, f32 = null;

try {
  const mod = new WebAssembly.Module(buildModule());
  const inst = new WebAssembly.Instance(mod, {});
  mm = inst.exports.matmul;
  f32 = new Float32Array(inst.exports.mem.buffer);
  USE_WASM = true;
} catch (e) {
  console.error("[wasm] 编译失败，回退纯 JS：", e.message);
}

// 纯 JS 参照实现（回退 + 自检基准）
function jsMatmul(A, B) {
  const n = A.length, k = B.length, m = B[0].length;
  const C = Array.from({ length: n }, () => new Float32Array(m));
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      const a = A[i][p]; if (a === 0) continue;
      const Bp = B[p], Ci = C[i];
      for (let j = 0; j < m; j++) Ci[j] += a * Bp[j];
    }
  return C;
}

function wasmMatmul(A, B) {
  const n = A.length, k = B.length, m = B[0].length;
  const aBase = A_OFF >> 2, bBase = B_OFF >> 2, cBase = C_OFF >> 2;
  // 三分区容量检查：A∈[0,4MB) B∈[4MB,10MB) C∈[10MB,16MB)，超出则回退 JS
  if (n * k * 4 > B_OFF - A_OFF || k * m * 4 > C_OFF - B_OFF || n * m * 4 > f32.byteLength - C_OFF)
    return jsMatmul(A, B);                             // 超出预留分区则回退
  for (let i = 0; i < n; i++) f32.set(A[i], aBase + i * k);
  for (let p = 0; p < k; p++) f32.set(B[p], bBase + p * m);
  f32.fill(0, cBase, cBase + n * m);                   // C 必须清零（核内是累加）
  mm(A_OFF, B_OFF, C_OFF, n, k, m);
  const C = new Array(n);
  for (let i = 0; i < n; i++) C[i] = f32.slice(cBase + i * m, cBase + (i + 1) * m);
  return C;
}

// ---------- 自检：与 JS 实现逐元素比对 ----------
function selfTest() {
  if (!USE_WASM) return false;
  const rand = (r, c) => Array.from({ length: r }, () =>
    Float32Array.from({ length: c }, () => Math.random() * 2 - 1));
  for (const [n, k, m] of [[26, 192, 768], [26, 768, 192], [5, 192, 192], [26, 384, 1536], [26, 1536, 384], [384, 26, 1536]]) {
    const A = rand(n, k), B = rand(k, m);
    const C1 = jsMatmul(A, B), C2 = wasmMatmul(A, B);
    let maxDiff = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) maxDiff = Math.max(maxDiff, Math.abs(C1[i][j] - C2[i][j]));
    if (maxDiff > 1e-3) { console.error(`[wasm] 自检失败 ${n}×${k}×${m} 最大误差 ${maxDiff}`); return false; }
  }
  return true;
}

// 若自检不过，禁用 wasm，全部回退 JS（保证正确性 > 性能）
if (USE_WASM && !selfTest()) USE_WASM = false;

// 矩阵转置（反向传播需要 Bᵀ、Aᵀ）
function transpose(M) {
  const r = M.length, c = M[0].length;
  const T = Array.from({ length: c }, () => new Float32Array(r));
  for (let i = 0; i < r; i++) { const Mi = M[i]; for (let j = 0; j < c; j++) T[j][i] = Mi[j]; }
  return T;
}

module.exports = {
  USE_WASM,
  matmul: (A, B) => (USE_WASM ? wasmMatmul(A, B) : jsMatmul(A, B)),
  transpose,
  jsMatmul,
  selfTest,
};
