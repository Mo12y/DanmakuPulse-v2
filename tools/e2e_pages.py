"""e2e_pages.py — B2.3 页面 E2E 关键路径（Windows 侧 Playwright → VM nginx :8088）

检查：3 页均可打开、无 console error / pageerror、关键元素渲染（卡片/表格/图表 canvas）、
     截图留证 → .workbuddy/outputs/e2e_*.png
运行：MSYS_NO_PATHCONV=1 "C:/Users/m1309/AppData/Local/Programs/Python/Python313/python.exe" e2e_pages.py
"""
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = 'http://192.168.88.100:8088'
OUT = r'D:\大学\new\.workbuddy\outputs'

CHECKS = {
    'overview.html': {
        'wait': '#hotTable tbody tr',
        'expect': [('#cards .card', 4), ('#hotTable tbody tr', 10), ('canvas', 2)],
    },
    'rhythm.html': {
        'wait': '#curve canvas',
        'expect': [('canvas', 3), ('#badges .badge', 1)],
    },
    'partition.html': {
        'wait': '#vidTable tbody tr',
        'expect': [('#baseCurve canvas', 1), ('#vidTable tbody tr', 5), ('#featTable tbody tr', 10)],
    },
}

results = {}
errors_total = []
with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    console_errors = []
    page.on('console', lambda m: console_errors.append(f'{m.type}: {m.text}')
            if m.type in ('error',) else None)
    page.on('pageerror', lambda e: console_errors.append(f'pageerror: {e}'))
    for name, spec in CHECKS.items():
        console_errors.clear()
        url = f'{BASE}/{name}'
        try:
            page.goto(url, wait_until='networkidle', timeout=45000)
            try:
                page.wait_for_selector(spec['wait'], timeout=20000)
            except Exception:
                pass
            time.sleep(2.0)   # 等图表动画/二次请求
            counts = {}
            for sel, min_n in spec['expect']:
                n = page.locator(sel).count()
                counts[sel] = n
            ok = all(counts[sel] >= min_n for sel, min_n in spec['expect'])
            shot = fr'{OUT}\e2e_{name.replace(".html", "")}.png'
            page.screenshot(path=shot, full_page=False)
            results[name] = {'ok': ok, 'counts': counts,
                             'expected': {s: m for s, m in spec['expect']},
                             'console_errors': list(console_errors), 'shot': shot}
            if not ok:
                errors_total.append(f'{name}: 元素计数未达标 {counts}')
            if console_errors:
                errors_total.append(f'{name}: console errors {console_errors}')
        except Exception as e:
            results[name] = {'ok': False, 'error': repr(e)}
            errors_total.append(f'{name}: {e!r}')
    browser.close()

print(json.dumps(results, ensure_ascii=False, indent=2))
print('=' * 60)
if errors_total:
    print('❌ E2E 发现问题：')
    for e in errors_total:
        print(' -', e)
    sys.exit(1)
print('✅ 3 页 E2E 关键路径全过（无 console error，关键元素渲染齐备）')
