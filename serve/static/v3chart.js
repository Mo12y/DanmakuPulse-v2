/* v3chart.js —— 图表取色助手（所有页面共用）
 *
 * 为什么需要它：ECharts 画在 canvas 上，**不认 CSS 变量**。
 * 直接在 option 里写 'var(--accent)' 会静默失效（本项目踩过一次：包络色消失）。
 * DESIGN.md §1#1 规定：图表用 dpVar('--token', '#fallback')，
 * 且 fallback 必须与该 token 的浅色值**逐字一致**（否则 CSS 未加载时会闪错色）。
 * 这条由 tests/test_v3_design_tokens.py 核对。
 */
(function (w) {
  function dpVar(name, fallback) {
    try {
      var v = getComputedStyle(document.documentElement).getPropertyValue(name);
      v = (v || '').trim();
      return v || fallback;
    } catch (e) { return fallback; }
  }
  w.dpVar = dpVar;

  /* 一张图里各系列的角色 → token（DESIGN.md §1.3：状态色绝不上数据图形）
     你（数据主体）   --primary
     同类中位/对标对象（参照系） --chart-axis（靠线型区分：虚线 vs 点线）
     偏离段（该看的地方） --accent-tint；峰值 --accent
     普遍范围带（包络） --chart-fill-primary
  */
  w.dmChart = function () {
    return {
      primary: dpVar('--primary', '#4A7DDF'),
      axis: dpVar('--chart-axis', '#706D66'),
      split: dpVar('--chart-split', '#F0ECE3'),
      accent: dpVar('--accent', '#C93D68'),
      accentTint: dpVar('--accent-tint', 'rgba(201,61,104,0.06)'),
      band: dpVar('--chart-fill-primary', 'rgba(74,125,223,0.08)'),
      tipBg: dpVar('--chart-tip-bg', '#FFFDF9'),
      tipBorder: dpVar('--chart-tip-border', '#D8D1C1'),
      text: dpVar('--text', '#292723'),
      textTertiary: dpVar('--text-tertiary', '#6C6760'),
      font: dpVar('--font', '-apple-system, "PingFang SC", "Microsoft YaHei", "Helvetica Neue", sans-serif')
    };
  };
})(window);
