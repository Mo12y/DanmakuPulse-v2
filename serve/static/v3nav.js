/* v3nav.js —— 统一顶栏（所有页面共用）
 *
 * 契约：
 *   · 页面放 <div class="topbar" id="v3top"></div> 并引入本文件
 *   · window.V3.ready(fn)  → /api/me 拿到后回调 fn(me)（未登录 me=null）
 *   · window.V3.onTheme = fn  → 切主题后回调（图表页用来重画）
 *   · window.V3.toast(msg) / window.V3.esc(s) / window.V3.markUrl()
 *
 * 导航只留用户真正会用的四页 + 管理员一页（DESIGN.md §5.1：导航不是目录树）。
 * rhythm.html / index.html 两个内部页 2026-10-16 下线，入口一并移除。
 * 管理员入口只对管理员渲染；真正门禁在 nginx auth_request + FastAPI（前端隐藏只是体验）。
 */
(function () {
  var THEME_KEY = 'v3-theme';
  var PAGES = [
    ['diagnosis.html', '视频诊断卡', false],
    ['overview.html', '市场总览', false],
    ['partition.html', '分区画像', false],
    ['me.html', '我的收藏', false],
    ['users.html', '用户管理', true]
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
  // 默认深色：用户指定的「v2 那版」观感就是深色（2026-10-16 对齐）。
  // 切过之后以用户的选择为准（localStorage）。浅色主题（暖米灰）完整保留，点 ◐ 可切。
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
    var el = document.getElementById('v3top');
    if (!el) return;
    var isAdmin = !!(me && me.role === 'admin');
    var cur = here();
    var links = PAGES.filter(function (p) { return !p[2] || isAdmin; }).map(function (p) {
      var on = p[0] === cur;
      /* §7.1：当前页不只靠颜色，加 aria-current */
      return '<a href="' + p[0] + '"' + (on ? ' class="on" aria-current="page"' : '') + '>' + p[1] + '</a>';
    }).join('');
    var user = me
      ? '<span class="who">' + V3.esc(me.username) + '<span class="tag' + (isAdmin ? ' admin' : '') + '">' +
        (isAdmin ? '管理员' : '用户') + '</span></span><a class="btn" href="/logout">退出</a>'
      : '<a class="btn" href="/login">登录</a>';
    el.innerHTML = '<div class="brand"><span class="dot"></span>弹幕节奏 · 对标诊断</div>' +
      '<nav class="nav" aria-label="主导航">' + links + '</nav>' + user +
      '<button class="btn" id="v3theme" type="button" aria-label="切换深浅色" title="切换深浅色">◐</button>';
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
  // cache:'no-store' 与服务端 Cache-Control: no-store 双保险 ——
  // 曾经因为 /api/me 被缓存 5 分钟，换账号后顶栏还是旧用户
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
