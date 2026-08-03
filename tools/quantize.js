// ============================================================
// quantize.js —— 把 f32 权重量化成 int8（只为了传输体积）
//
// 动机是产品问题：浏览器本地模式要下 210MB，那是「演示一下」的量级；
// 53MB 才是「你可以自己用」。
//
// 方案：逐行（逐输出通道）对称量化
//   scale[r] = max|W[r,:]| / 127
//   q[r,c]   = round(W[r,c] / scale[r])       存 int8
//   还原     W ≈ q * scale[r]
// 逐行而非全张量共享一个 scale：不同输出通道的幅度能差一两个数量级，
// 共享 scale 会让小幅度的行几乎全部塌到 0。
//
// 小张量（LayerNorm 的 gain/bias、matmul 的 bias，共 42 个 / 105KB）保持 f32：
// 量化它们省不到 0.1MB，却直接动到归一化的数值稳定性，不划算。
//
// 在加载时还原成 f32 再上传显存，所以推理内核一行都不用改 —— 显存占用不变，
// 省的是下载。要连显存一起省得让每个 matmul 内核就地反量化，那是另一回事。
//
// 用法: deno run --allow-read --allow-write tools/quantize.js poet-weights-v3-best
// ============================================================
const SMALL = 10000;          // 元素数 ≤ 此值的张量不量化

const src = Deno.args[0] || "poet-weights-v3-best";
const dst = Deno.args[1] || `${src}-i8`;
const wp = (f) => new URL(`../weights/${f}`, import.meta.url);

const meta = JSON.parse(Deno.readTextFileSync(wp(`${src}.meta.json`)));
if (meta.format !== "bin-f32") throw new Error(`只能量化 bin-f32，实际是 ${meta.format}`);
const raw = new Float32Array(Deno.readFileSync(wp(`${src}.bin`)).buffer);

let off = 0;
const q8 = [], scales = [], keepF32 = [], outTensors = [];
let maxRelErr = 0, sumSqErr = 0, sumSq = 0, nQ = 0;

for (const t of meta.tensors) {
  const n = t.rows * t.cols;
  const w = raw.subarray(off, off + n);
  off += n;
  if (n <= SMALL) {
    keepF32.push(w);
    outTensors.push({ ...t, q: false });
    continue;
  }
  const q = new Int8Array(n);
  const sc = new Float32Array(t.rows);
  for (let r = 0; r < t.rows; r++) {
    const base = r * t.cols;
    let mx = 0;
    for (let c = 0; c < t.cols; c++) { const a = Math.abs(w[base + c]); if (a > mx) mx = a; }
    const s = mx / 127 || 1e-12;          // 整行为 0 时给个非零 scale，避免还原时出 NaN
    sc[r] = s;
    for (let c = 0; c < t.cols; c++) {
      const v = Math.max(-127, Math.min(127, Math.round(w[base + c] / s)));
      q[base + c] = v;
      const err = w[base + c] - v * s;
      sumSqErr += err * err; sumSq += w[base + c] * w[base + c]; nQ++;
      const rel = mx > 0 ? Math.abs(err) / mx : 0;
      if (rel > maxRelErr) maxRelErr = rel;
    }
  }
  q8.push(q); scales.push(sc);
  outTensors.push({ ...t, q: true });
}
if (off !== raw.length) throw new Error(`张量总长与文件不符: ${off} vs ${raw.length}`);

// 落盘布局：[所有 int8 块][所有 scale f32 块][所有未量化的 f32 块]
// 三段各自按 meta.tensors 顺序排列，加载时按同样顺序切回来。
const bytesI8 = q8.reduce((s, a) => s + a.length, 0);
const bytesSc = scales.reduce((s, a) => s + a.length * 4, 0);
const bytesF32 = keepF32.reduce((s, a) => s + a.length * 4, 0);
const buf = new Uint8Array(bytesI8 + bytesSc + bytesF32);
let p = 0;
for (const a of q8) { buf.set(new Uint8Array(a.buffer, a.byteOffset, a.length), p); p += a.length; }
for (const a of scales) { buf.set(new Uint8Array(a.buffer, a.byteOffset, a.length * 4), p); p += a.length * 4; }
for (const a of keepF32) { buf.set(new Uint8Array(a.buffer, a.byteOffset, a.length * 4), p); p += a.length * 4; }

Deno.writeFileSync(wp(`${dst}.bin`), buf);
Deno.writeTextFileSync(wp(`${dst}.meta.json`), JSON.stringify({
  ...meta,
  format: "bin-i8-rowsym",
  quantizedFrom: src,
  smallThreshold: SMALL,
  tensors: outTensors,
}));

const mb = (b) => (b / 1048576).toFixed(1);
console.log(`${src} → ${dst}`);
console.log(`  int8 ${mb(bytesI8)}MB + scale ${mb(bytesSc)}MB + 未量化 f32 ${mb(bytesF32)}MB = ${mb(buf.length)}MB`);
console.log(`  原始 ${mb(raw.length * 4)}MB，压到 ${(100 * buf.length / (raw.length * 4)).toFixed(1)}%`);
console.log(`  量化了 ${nQ.toLocaleString()} 个参数，相对均方误差 ${(Math.sqrt(sumSqErr / sumSq) * 100).toFixed(3)}%，单点最大相对误差 ${(maxRelErr * 100).toFixed(3)}%`);
