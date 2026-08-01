# mini-gpt-js

纯 JavaScript 从零实现的 GPT（nanoGPT 风格），一路从玩具做到 GPU 全栈：

```
纯 JS 手写 Transformer → Worker 数据并行 → WASM SIMD 矩阵乘 → WebGPU 内核 → 全 GPU 训练
```

目标是训练一个「唐宋诗词生成器」：以 9,064 个汉字为词表，10 层 / 640 维 / 10 头注意力，
在 ~30MB 唐宋诗词语料上训练，权重以 `f32 .bin + meta.json` 格式存取（各模块互通）。

## 亮点

- **零依赖**：没有 PyTorch / NumPy / ONNX，全部矩阵运算手写
- **从 CPU 到 GPU 的完整演进路径**，每一步都有可运行的版本
- **全算子 WGSL 化**：embed / LayerNorm / 多头因果注意力 / GELU / matmul(NN/NT/TN) /
  crossEntropy / Adam 全部为 WebGPU compute shader；权重、梯度、Adam 状态常驻显存，
  每步 CPU 只上传 batch 的 token id
- **一份推理代码两端运行**：Deno 服务端与浏览器（`<script type="module">`）共用
  `webgpu-forward.js`，浏览器可下载权重后完全本地推理
- **单一数据源**：语料、词表、训练/验证切分统一由 `data-split.js` 提供，杜绝各处手抄不一致
- **WASM 手写内核**：不用任何工具链，JS 内迷你汇编器直接吐 wasm 二进制，f32x4 SIMD 矩阵乘

## 演进路线（对应代码）

| 阶段 | 文件 | 内容 |
|---|---|---|
| 1. 前向 | `mini-gpt.js` | 纯 JS GPT 前向 + 文本生成（结构与 GPT-2 / nanoGPT 一致） |
| 2. 训练 | `mini-gpt-train.js` | 纯 JS 反向传播 + Adam，`node mini-gpt-train.js` 直接跑 |
| 3. 可用 | `mini-gpt-poet.js` | 古诗灵感生成器，首次运行自动训练并存盘权重 |
| 4. 并行 | `mini-gpt-parallel.js` | Worker 数据并行训练，8 Worker 梯度求和，数学上等价于串行 |
| 5. WASM | `wasm-kernel.js` | 手写 WebAssembly SIMD（f32x4）矩阵乘，带数值对齐自检 |
| 6. WebGPU 内核 | `webgpu-kernel.js` | WGSL matmul 内核 + 分档基准测试 |
| 7. 全 GPU 训练 | `gpu-train.js` | Deno + WebGPU，全部算子 WGSL 化，b=32 首诗 800 token/步 |
| 8. GPU 推理 | `webgpu-forward.js` | 推理模块，Deno 服务端与浏览器共用 |
| 9. 服务 | `gpu-server.js` / `server.js` | GPU 版 / 零依赖 Node 版古诗生成 Web 服务 |

## 模型配置

| 参数 | 值 |
|---|---|
| vocabSize | 9,064（语料去重后的汉字） |
| nLayer / nHead / nEmbd | 10 / 10 / 640 |
| blockSize | 66（最长七律的 token 数，短诗 padding + 掩码） |
| batch | 32 首诗 × 65 token |
| 语料 | 唐宋诗词 ~30MB（`poems-tangsong.txt`） |

最新 checkpoint（v3）：40 万步，train loss **4.03** / val loss **4.24**，目标 200 万步，
训练速度约 21ms/样本（M 系列 Mac 上 WebGPU 实测）。

## 快速开始

```bash
# 纯 JS 玩转：零依赖，首次自动训练小模型
node mini-gpt-poet.js 月 5        # 以"月"开头生成 5 首

# 纯 JS 手动训练（CPU）
node mini-gpt-train.js

# Worker 数据并行训练（8 核）
node mini-gpt-parallel.js

# WASM SIMD 内核自检 + 基准
node wasm-kernel.js

# 全 GPU 训练（Deno + WebGPU，需要支持 WebGPU 的环境）
deno run --allow-read --allow-write gpu-train.js 2000000

# GPU 古诗生成服务
deno run --no-code-cache --allow-read --allow-net gpu-server.js 8888
# 浏览器打开 http://localhost:8888 ，页面提供「本地模式」：权重下载到浏览器后断网可作诗

# 零依赖 Node 服务
node server.js 8888
```

## 权重格式

- `poet-weights-*.bin`：f32 张量顺序拼接（meta.json 中 `tensors` 记录了每张量名称与行列）
- `poet-weights-*.meta.json`：配置、词表、tensor 布局、训练/验证 loss、语料来源
- `poet-weights-*-curve.csv`：训练曲线（step, trainLoss, valLoss）

`gpu-train.js` 产出的权重可直接被 `mini-gpt-poet.js` / `webgpu-forward.js` 加载推理或续训。

> 注意：权重文件体积超 GitHub 100MB 单文件限制，已通过 `.gitignore` 排除。
> 用 `gpu-train.js` 从零训练，或直接跑 `mini-gpt-poet.js` 自动训练小模型即可复现。

## 辅助脚本

- `fetch-poems*.js`：语料抓取与清洗
- `data-split.js`：语料 / 词表 / 训练验证切分（唯一来源）
- `compare-models.js` / `eval-model.js`：模型对比与评估
- `probe-memory.js` / `probe-song.js`：内部表征探测（词向量、注意力等）
- `verify-corpus.js`：语料合法性校验
- `test-forward.js` / `test-kv.js`：前向与 KV cache 测试
