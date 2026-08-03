// ============================================================
// normalize.js —— 输入归一化与逐字体检（服务端与浏览器共用）
//
// 映射表（t2s-map.json）与词表都由调用方注入，所以这个模块在 Deno 与
// 浏览器里都能跑：服务端读文件，浏览器 fetch。表是单一来源，
// 应用逻辑也只写一遍——两边各抄一份的话，某天只改一边就会出现
// 「服务端接受、本地模式拒绝」这种说不清的差异。
// ============================================================

// 逐字体检：返回每个字的诊断，供前端标色、供服务端判断能否生成
//   raw       用户输入的原字
//   use       归一化后实际送进模型的字
//   ok        归一化后是否在该模型词表里
//   converted 是否发生了繁简转换（前端用来显示「舉→举」）
export function makeChecker(t2s, stoi) {
  return (text) =>
    [...text].map((raw) => {
      const use = t2s[raw] || raw;
      return { raw, use, ok: use in stoi, converted: use !== raw };
    });
}

// 送进模型的最终文本：丢掉词表里没有的字
// 这是「静默改写用户输入」，所以只在用户已经看到标红、明确点了继续时才用
export const usable = (report) => report.filter((x) => x.ok).map((x) => x.use).join("");

// 词表里没有的原字（去重，保持出现顺序）
export const missing = (report) => [...new Set(report.filter((x) => !x.ok).map((x) => x.raw))];
