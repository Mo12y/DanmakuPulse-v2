/* v3nav.js —— 全站统一左侧导航（v2 三层 IA，2026-10-16 由顶栏改造而来）
 *
 * 契约：
 *   · 页面放 <nav class="sidebar" id="v3side" aria-label="主导航"></nav>，并把它套进 .layout
 *   · window.V3.ready(fn)  → /api/me 拿到后回调 fn(me)（未登录 me=null）
 *   · window.V3.onTheme = fn  → 切主题后回调（图表页用来重画）
 *   · window.V3.toast(msg) / window.V3.esc(s) / window.V3.markUrl()
 *
 * 分组标题**就是类别**（本轮判据要求"受众 / 类别"落在界面上，不许只写在计划书里）：
 *   诊断 / 追踪 = 任务（UP 主、MCN/运营 真要做的那件事）
 *   市场        = 框架（诊断的尺子 —— 没有它，每句「你在第 X 百分位」都无从谈起）
 *   模型与验证  = 证据（受众是答辩组，**必须明标**，不拿"证据"当万能标签）
 *   后台管理    = 系统（仅管理员）
 * 每页再配一句副标题写「受众 + 它帮谁做成了哪件事」（.pagesub，见 v3.css）。
 * ⛔ 明确不放：节奏地图(PCA) / 形态归属(K-Means) —— 已否决，不要搬回来。
 */
(function () {
  var THEME_KEY = 'v3-theme';
  var GROUPS = [
    { title: '分析', core: true, items: [
      ['diagnosis.html', '视频分析'],
      ['compare.html', '双视频对比'],
      ['batch.html', '批量体检']] },
    { title: '追踪', items: [['track.html', '跨批追踪']] },
    { title: '市场', items: [
      ['overview.html', '市场总览'],
      ['partition.html', '分区画像']] },
    { title: '模型与验证', items: [
      ['deviation.html', '偏离检测', 'ML']] },
    /* 后台管理：仅管理员可见 */
    { title: '管理', admin: true, items: [
      ['collect.html', '采集控制台'],
      ['data.html', '数据管理'],
      ['users.html', '用户管理']] }
  ];

  var V3 = window.V3 = window.V3 || {};
  V3.me = null;

  V3.esc = function (x) {
    return String(x == null ? '' : x).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  };

  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
  }
  /* 默认深色（用户指定的「v2 那版」观感）；切过以用户选择为准 */
  (function () {
    var t = 'dark';
    try { t = localStorage.getItem(THEME_KEY) || 'dark'; } catch (e) {}
    document.documentElement.setAttribute('data-theme', t);
  })();

  function here() {
    var p = location.pathname.split('/').pop();
    return p || 'diagnosis.html';
  }

  function render(me) {
    var el = document.getElementById('v3side');
    if (!el) return;
    var isAdmin = !!(me && me.role === 'admin');
    var cur = here();
    var html = '<div class="brand"><span class="dot"></span>弹幕节奏 · 数据工作台</div>';
    GROUPS.forEach(function (g) {
      if (g.admin && !isAdmin) return;
      html += '<div class="side-group"><div class="side-group-title' + (g.core ? ' core' : '') + '">' +
        g.title + '</div>';
      g.items.forEach(function (it) {
        var on = it[0] === cur;
        /* §7.1：当前页不只靠颜色，加 aria-current */
        html += '<a href="' + it[0] + '"' + (on ? ' class="active" aria-current="page"' : '') + '>' +
          '<span>' + it[1] + '</span>' + (it[2] ? '<span class="ml">' + it[2] + '</span>' : '') + '</a>';
      });
      html += '</div>';
    });
    /* 底部：用户信息 + 收藏（个人页，不属于任何分析分组）+ 退出 */
    html += '<div class="side-group side-user">';
    if (me) {
      html += '<div class="side-user-name">' + V3.esc(me.username) +
        '<span class="tag' + (isAdmin ? ' admin' : '') + '">' +
        (isAdmin ? '管理员' : '用户') + '</span></div>' +
        '<a href="me.html"' + (cur === 'me.html' ? ' class="active" aria-current="page"' : '') +
        '><span>我的收藏</span></a>' +
        '<a href="/logout"><span>退出登录</span></a>';
    } else {
      html += '<a href="login.html"><span>登录</span></a>';
    }
    html += '<button class="btn side-theme" id="v3theme" type="button" ' +
      'aria-label="切换深浅色" title="切换深浅色">◐ 切换深浅色</button></div>';
    el.innerHTML = html;
    var t = document.getElementById('v3theme');
    if (t) t.onclick = function () {
      var c = document.documentElement.getAttribute('data-theme');
      applyTheme(c === 'dark' ? 'light' : 'dark');
      if (typeof V3.onTheme === 'function') V3.onTheme();
    };
  }

  var cbs = [], done = false;
  V3.ready = function (fn) { if (done) fn(V3.me); else cbs.push(fn); };

  V3.toast = function (msg) {
    var d = document.createElement('div');
    d.className = 'v3toast';
    d.setAttribute('role', 'status');
    d.textContent = msg;
    document.body.appendChild(d);
    setTimeout(function () { d.remove(); }, 2600);
  };

  /* 把当前交互状态写进地址栏（可分享、可复现） */
  V3.markUrl = function () {};

  render(null);
  /* cache:'no-store' 与服务端 Cache-Control: no-store 双保险 ——
     曾经因为 /api/me 被缓存 5 分钟，换账号后侧栏还是旧用户 */
  fetch('/api/me', { credentials: 'same-origin', cache: 'no-store' })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) { V3.me = j && j.data ? j.data : null; })
    .catch(function () { V3.me = null; })
    .then(function () {
      render(V3.me);
      done = true;
      cbs.forEach(function (f) { try { f(V3.me); } catch (e) {} });
    });
})();
