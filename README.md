# mini-gpt-js

纯 JavaScript 从零实现的 GPT（nanoGPT 风格），一路从玩具做到 GPU 全栈：

```
纯 JS 手写 Transformer → Worker 数据并行 → WASM SIMD 矩阵乘 → WebGPU 内核 → 全 GPU 训练
```

**🖋 在线体验：<https://exposir.github.io/mini-gpt-js/>**
（模型在你浏览器的 GPU 里运行——首次下载 53MB 权重，之后断网可用。需要 Chrome/Edge 113+ 或 Safari 26+）

目标是训练一个「唐宋诗词生成器」：以 9,064 个汉字为词表，10 层 / 640 维 / 10 头注意力，
在 22.4 万首唐宋诗（1009 万字）上训练；支持五绝/七绝/五律/七律体裁约束与自动押韵，
逐字流式输出，可导出竖排诗笺图片。

## 亮点

- **零依赖**：没有 PyTorch / NumPy / ONNX，全部矩阵运算手写
- **从 CPU 到 GPU 的完整演进路径**，每一步都有可运行的版本
- **全算子 WGSL 化**：embed / LayerNorm / 多头因果注意力 / GELU / matmul(NN/NT/TN) /
  crossEntropy / Adam 全部为 WebGPU compute shader；权重、梯度、Adam 状态常驻显存，
  每步 CPU 只上传 batch 的 token id
- **一份推理代码两端运行**：Deno 服务端与浏览器（`<script type="module">`）共用
  `webgpu-forward.js`，浏览器可下载权重后完全本地推理
- **int8 量化只为传输**：逐行对称量化 210MB→53MB（25.2%），加载时反量化、内核零改动；
  同一段 eval 代码验收：val loss 4.0383 → 4.0385，差 0.0002，低于采样标准误 0.011
- **格律在解码层实现，不重训**：字数句数由采样 mask 强制；押韵表不用韵书，
  从 22.4 万首近体诗的偶数句尾字共现统计自举（学到唐人实际通押，如「涯」押支韵）
- **单一数据源**：语料、词表、切分、校验、韵表、解包各只写一份，杜绝手抄不一致
- **WASM 手写内核**：不用任何工具链，JS 内迷你汇编器直接吐 wasm 二进制，f32x4 SIMD 矩阵乘

## 目录结构

```
gpu/      主线：WebGPU 训练与推理
  webgpu-forward.js   推理（服务端与浏览器共用这一份；体裁/押韵 mask 也在这）
  webgpu-kernel.js    WGSL matmul 内核 + 基准测试
  gpu-train.js        全算子 WGSL 化的训练器（eval 模式兼作量化验收）
  gpu-server.js       Web 服务（三模型可切，SSE 逐字流式）
  load-model.js       按前缀加载快照（字表随模型走的规则在这里统一）
  unpack-weights.js   f32 / int8 双格式解包（推理与评估共用）
cpu/      项目起点：纯 JS / Worker / WASM 实现
data/     语料、词表、训验切分；繁简映射与押韵表的生成器
docs/     静态站（GitHub Pages）：纯浏览器单模型版，由 tools/build-site.js 同步
tools/    出诗、对拍、记忆探针、打分、扫参、量化、体裁回归、建站
weights/  *.bin + *.meta.json + *-curve.csv
logs/     训练与服务日志
```

## 演进路线（对应代码）

| 阶段 | 文件 | 内容 |
|---|---|---|
| 1. 前向 | `cpu/mini-gpt.js` | 纯 JS GPT 前向 + 文本生成（结构与 GPT-2 / nanoGPT 一致） |
| 2. 训练 | `cpu/mini-gpt-train.js` | 纯 JS 反向传播 + Adam |
| 3. 可用 | `cpu/mini-gpt-poet.js` | 古诗灵感生成器，首次运行自动训练并存盘权重 |
| 4. 并行 | `cpu/mini-gpt-parallel.js` | Worker 数据并行训练，8 Worker 梯度求和，数学上等价于串行 |
| 5. WASM | `cpu/wasm-kernel.js` | 手写 WebAssembly SIMD（f32x4）矩阵乘，带数值对齐自检 |
| 6. WebGPU 内核 | `gpu/webgpu-kernel.js` | WGSL matmul 内核 + 分档基准测试 |
| 7. 全 GPU 训练 | `gpu/gpu-train.js` | Deno + WebGPU，全部算子 WGSL 化，b=32 首诗 800 token/步 |
| 8. GPU 推理 | `gpu/webgpu-forward.js` | 推理模块，Deno 服务端与浏览器共用 |
| 9. 服务 | `gpu/gpu-server.js` / `cpu/server.js` | GPU 版 / 零依赖 Node 版古诗生成 Web 服务 |
| 10. 产品化 | `tools/quantize.js` / `data/build-rhyme.js` / `docs/` | int8 量化、押韵表自举、纯浏览器静态站上线 |

## 模型配置

| 参数 | 值 |
|---|---|
| vocabSize | 9,064（语料去重后的汉字） |
| nLayer / nHead / nEmbd | 10 / 10 / 640 |
| blockSize | 66（最长七律的 token 数，短诗 padding + 掩码） |
| batch | 32 首诗 × 65 token |
| 语料 | 唐宋诗词 ~30MB（`poems-tangsong.txt`） |

训练结果（v3，1009 万字唐宋诗）：跑满 200 万步，**验证集拐点在 110 万步，val loss 4.0383**；
终态 val 4.4130 / train 2.8210（已过拟合）。训练速度约 21ms/样本（M 系列 Mac 上 WebGPU 实测）。

两炉对照给出一个意外观测：语料从 157 万字扩到 1009 万字（×6.4），**过拟合拐点依旧落在第 5 个 epoch**
（分别是 5.04 与 4.95）—— 扩数据买到的不是“能训更久”，而是“每个 epoch 更值钱”：
val loss 4.5109 → 4.0383，模型的候选字数从 e^4.51≈91 降到 e^4.04≈57。

推理带重复惩罚（已出现的字扣减 logits，标点与换行预留），定稿强度 2.0：
重复率 8.7% → 1.5%（真诗基线 1.9%），结构合法率 97% → 100%。

## 格律是怎么实现的

模型自由生成时会混 4/5/6/7 字句、押韵靠自发——**格律全部在解码层用采样 mask 强制，不动模型**：

- **字数句数**：字数未满时把标点 logits 打成 -∞；满了直接 emit 该行标点（奇句「，」偶句「。」）
- **押韵**：第一个韵脚（第 2 句尾字）在「可押韵字集」内采样定锚，
  后续偶数句尾字只允许锚的通押邻居（剔除已用韵脚防重韵）；允许集为空则回退自由采样，
  宁可偶尔出韵，不会卡死
- **韵表**（`data/rhyme-map.json`，194KB）不抄韵书，由 `data/build-rhyme.js` 从语料自举：
  取 22.4 万首结构规整的近体诗，统计偶数句尾字的两两共现（≥3 次为强边），
  得 3,100 字 / 2.85 万条通押边。模型从哪份语料学的，约束就来自哪份语料——
  学到的是唐人「实然」通押（支微、鱼虞，「涯」押支韵），而非韵书「应然」

繁简归一化同理：语料是繁转简洗过的，词表只有简体，`data/build-t2s.js` 用 opencc-js
预生成 2,942 对单字映射（34KB 静态表，运行时零依赖），输入端逐字体检、生僻字即时标红。

## 快速开始

```bash
# 出诗（默认用验证集最优的 v3-best）
npm run gen -- "断桥是否下过雪" 6

# Web 服务（三模型可切：当前最优 / 小语料对照 / 过拟合标本）
npm run serve
# 浏览器打开 http://localhost:8890 ，页面提供「本地模式」：权重下载到浏览器后断网可作诗

# 全 GPU 训练（第 2、3 个参数是权重前缀与评估间隔）
npm run train -- 2000000 poet-weights-v3 50000

# 评估三把尺子
npm run compare                      # 同题对拍：原创率 / 重复率 / 用字广度
npm run probe                        # 记忆探针：拿留出诗首句看它会不会背原作
npm run tune                         # 扫重复惩罚强度

# 量化与静态站
deno run --allow-read --allow-write tools/quantize.js poet-weights-v3-best   # f32 → int8（53MB）
deno run --no-code-cache --allow-read --allow-env gpu/gpu-train.js eval      # 量化前后 val loss 验收
deno run --no-code-cache --allow-read --allow-env tools/test-form.js         # 体裁+押韵回归断言
deno run --allow-read --allow-write tools/build-site.js                      # 同步 docs/ 静态站

# 项目起点：纯 JS，零依赖，首次自动训练小模型
npm run cpu:gen -- 月 5
npm run cpu:serve

# 重建语料（唐+宋 313 卷，产出 22.4 万首）
npm run fetch
```

## 权重格式

- `weights/poet-weights-*.bin`：f32 张量顺序拼接（meta.json 中 `tensors` 记录了每张量名称与行列）
- `weights/poet-weights-*-i8.bin`：int8 逐行对称量化（`[int8 块][scale f32][未量化小张量]`，
  LayerNorm 的 gain/bias 等 ≤10,000 元素的张量保留 f32），`unpack-weights.js` 统一解包
- `weights/poet-weights-*.meta.json`：配置、词表、tensor 布局、训练/验证 loss、语料来源
- `weights/poet-weights-*-curve.csv`：训练曲线（step, trainLoss, valLoss）

`gpu/gpu-train.js` 产出的权重可直接被 `cpu/mini-gpt-poet.js` / `gpu/webgpu-forward.js` 加载推理或续训。

> 注意：`weights/` 下的 f32 权重（210MB）超 GitHub 100MB 单文件限制，已通过 `.gitignore` 排除；
> 但量化后的 53MB 权重以 `docs/model.bin` 入库（静态站要用）——克隆仓库即得可推理的权重。
> 想从零复现就 `npm run train`，或跑 `npm run cpu:gen` 自动训练小模型。

## 辅助脚本

- `data/fetch/fetch-poems*.js`：语料抓取与清洗（v3 拓到唐+宋）
- `data/data-split.js`：语料 / 词表 / 训练验证切分（唯一来源，固定种子）
- `data/normalize.js`：输入校验（繁转简 + 词表体检），服务端与浏览器共用同一份
- `data/build-t2s.js` / `build-rhyme.js`：繁简映射表、押韵表生成器（产物入库，运行时零依赖）
- `data/fetch/verify-corpus.js` / `probe-song.js`：语料合法性校验、卷范围探测
- `tools/quantize.js`：int8 逐行对称量化（210MB→53MB）
- `tools/build-site.js`：同步 docs/ 静态站（算法与数据以 gpu/、data/、weights/ 为单一来源）
- `tools/compare-models.js`：同题对拍两个快照（原创率 / 重复率 / 用字广度）
- `tools/probe-memory.js`：记忆探针 —— temperature 压到 0.1 逼近贪心解码，能背就一定背出来
- `tools/score-model.js` / `tune-penalty.js`：快照打分、重复惩罚扫参
- `tools/test-forward.js` / `test-kv.js` / `test-form.js`：前向对拍、KV cache 对拍、体裁+押韵断言

## 一些踩过的坑

- **训练集 loss 分不清“真懂”与“背熟”**。必须留出验证集；上一炉靠人眼抽 12 首数原创率
  选的快照（35 万步），后来被双曲线证明已过拐点一半。
- **字表必须随模型走**。推理端若从语料现算字表，一换语料旧权重的 embedding 行号全错位，
  输出变乱码而**不报错**。现在字表存在 `meta.vocab` 里，并在 `createPoet` 里校长度。
- **词表外的字会静默变换行符**。`stoi[生僻字]` 是 undefined，塞进 Uint32Array 变 0（换行 token），
  模型收到的输入被悄悄改了——所以输入校验必须在生成之前拦住，而不是靠输出质量发现。
- **切分逻辑不能抄多份**。手抄错一个字符的后果是静默的（不报错，只给出错的结论）。
- **服务端内嵌页面的转义坑**。页面若以 JS 模板字符串内嵌，其中的 `"\n\n"` 会被外层转义成
  真换行、正则 `\w` 的反斜杠被吃掉——JS 字符串静默截断、SSE 解析静默失败。
  内嵌代码里要写 `\\n`，或干脆像 docs/ 那样用真 HTML 文件。
- **lr 调度会适得其反**。上一炉在 60% 处降 lr，train loss 从 2.32 陡降到 1.34 而 val 同步上翻
  —— 相当于把模型直接推进背书阶段。现在 lr 恒定。
- **指标差异要先测噪声底**。平台区 8 个候选的“差异”经控制实验（同一模型跑三遍）证明
  全是采样波动，不可区分。量化验收同理：i8 与 f32 的 val loss 差 0.0002，远低于采样标准误
  0.011，判定为无损。
