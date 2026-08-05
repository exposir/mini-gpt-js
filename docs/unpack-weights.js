// ============================================================
// unpack-weights.js —— 把权重文件解成「张量名 → Float32Array」
//
// 支持两种格式，调用方不用关心区别：
//   bin-f32        原始 f32 顺序拼接
//   bin-i8-rowsym  逐行对称量化的 int8 + 每行 scale，这里还原成 f32
//
// 为什么还原成 f32 而不让内核吃 int8：省的是下载体积（210MB→53MB），
// 显存与算术路径完全不变，推理内核一行都不用改。要连显存一起省，
// 得让每个 matmul 就地反量化，那是另一个量级的改动。
//
// 抽成单一来源是因为推理（webgpu-forward）与评估（gpu-train）都要读，
// 两边各写一份解包逻辑的话，量化格式的偏移量算错一处就会静默出乱码。
// ============================================================

export function unpackWeights(meta, binBuffer) {
  const slices = {};
  if (!meta.format || meta.format === "bin-f32") {
    const raw = new Float32Array(binBuffer);
    let off = 0;
    for (const t of meta.tensors) {
      slices[t.name] = raw.subarray(off, off + t.rows * t.cols);
      off += t.rows * t.cols;
    }
    if (off !== raw.length) throw new Error(`权重长度不符: 张量共 ${off}，文件 ${raw.length}`);
    return slices;
  }

  if (meta.format === "bin-i8-rowsym") {
    // 落盘布局：[全部 int8][全部 scale f32][全部未量化 f32]，三段各自按 tensors 顺序
    const qT = meta.tensors.filter((t) => t.q);
    const fT = meta.tensors.filter((t) => !t.q);
    const nI8 = qT.reduce((s, t) => s + t.rows * t.cols, 0);
    const nSc = qT.reduce((s, t) => s + t.rows, 0);
    const nF32 = fT.reduce((s, t) => s + t.rows * t.cols, 0);
    const expect = nI8 + (nSc + nF32) * 4;
    if (binBuffer.byteLength !== expect) {
      throw new Error(`i8 权重长度不符: 期望 ${expect}，实际 ${binBuffer.byteLength}`);
    }
    const i8 = new Int8Array(binBuffer, 0, nI8);
    // scale 与 f32 段起点未必 4 字节对齐（int8 段长度任意），先拷进对齐缓冲再读
    const tail = new Uint8Array(binBuffer, nI8, (nSc + nF32) * 4).slice();
    const sc = new Float32Array(tail.buffer, 0, nSc);
    const f32 = new Float32Array(tail.buffer, nSc * 4, nF32);

    let pI = 0, pS = 0;
    for (const t of qT) {
      const n = t.rows * t.cols;
      const out = new Float32Array(n);
      for (let r = 0; r < t.rows; r++) {
        const s = sc[pS + r], base = r * t.cols;
        for (let c = 0; c < t.cols; c++) out[base + c] = i8[pI + base + c] * s;
      }
      slices[t.name] = out;
      pI += n; pS += t.rows;
    }
    let pF = 0;
    for (const t of fT) {
      const n = t.rows * t.cols;
      slices[t.name] = f32.subarray(pF, pF + n);
      pF += n;
    }
    return slices;
  }

  throw new Error(`不认识的权重格式: ${meta.format}`);
}
