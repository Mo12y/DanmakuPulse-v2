/* v3 统一顶栏（所有页面共用，避免每页各写一份导航）
   军规：主视图零术语。管理员入口只对管理员渲染；真正的门禁在 nginx auth_request + FastAPI，
   前端隐藏只是体验（旧系统同款分工）。
   契约：
     · 页面放一个 <div class="topbar" id="v3top"></div> 并引入本文件
     · window.V3.ready(fn)  → /api/me 拿到后回调 fn(me)（未登录 me=null）
     · window.V3.onTheme = fn  → 切主题后回调（图表页用来重画）
     · window.V3.toast(msg)
     · window.V3.esc(s)
*/
(function () {
  var THEME_KEY = 'v3-theme';
  var PAGES = [
    ['diagnosis.html', '视频诊断卡', false],
    ['overview.html', '市场总览', false],
    ['partition.html', '分区画像', false],
    ['me.html', '我的收藏', false],
    ['rhythm.html', '原始细节', true],
    ['users.html', '用户管理', true],
    ['index.html', '验收页', false]
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
  try { var saved = localStorage.getItem(THEME_KEY); if (saved) document.documentElement.setAttribute('data-theme', saved); } catch (e) {}

  function here() {
    var p = location.pathname.split('/').pop();
    return p || 'index.html';
  }

  function render(me) {
    var el = document.getElementById('v3top');
    if (!el) return;
    var isAdmin = !!(me && me.role === 'admin');
    var cur = here();
    var links = PAGES.filter(function (p) { return !p[2] || isAdmin; }).map(function (p) {
      return '<a class="' + (p[0] === cur ? 'on' : '') + '" href="' + p[0] + '">' + p[1] + '</a>';
    }).join('');
    var user = me
      ? '<span class="who">' + V3.esc(me.username) + '<span class="tag' + (isAdmin ? ' admin' : '') + '">' +
        (isAdmin ? '管理员' : '用户') + '</span></span><a class="btn" href="/logout">退出</a>'
      : '<a class="btn" href="/login">登录</a>';
    el.innerHTML = '<div class="brand"><span class="dot"></span>弹幕节奏 · 对标诊断</div>' +
      '<div class="nav">' + links + '</div>' + user +
      '<button class="btn" id="v3theme" title="切换深浅色">◐</button>';
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
    d.textContent = msg;
    document.body.appendChild(d);
    setTimeout(function () { d.remove(); }, 2600);
  };

  render(null);   // 先按未登录渲染，避免闪烁整块空白
  fetch('/api/me', { credentials: 'same-origin' })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) { V3.me = j && j.data ? j.data : null; })
    .catch(function () { V3.me = null; })
    .then(function () {
      render(V3.me);
      done = true;
      cbs.forEach(function (f) { try { f(V3.me); } catch (e) {} });
    });
})();

