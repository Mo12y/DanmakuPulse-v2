# -*- coding: utf-8 -*-
"""collector.py — 采集控制台后端：真的能 start / stop 那套采集脚本

为什么能放进这个容器：collect/bili.py **只依赖 requests**，而且 batch_crawl 是
**逐个视频写盘**（save_batch_csv([v.copy()]) + manifest._save()），不在内存里累积 ——
所以这个容器跑得动（峰值远低于限值）。

代码与产物用 compose 挂进来（见 docker-compose.b0.yml 的 fastapi.volumes）：
    ../../danmaku/collect             -> /collect      (ro)  采集脚本
    ../../danmaku/data/cli_collected  -> /data-out     (rw)  产物 CSV
    ../../danmaku/logs                -> /logs         (rw)  采集日志

安全边界（这是从网页触发进程，必须写清楚）：
  · 只跑**一条固定命令**（collect/bili.py hotlist），参数以列表传递、**不过 shell**
  · 同一时刻只允许一个采集进程；重复 start 直接被拒
  · count 限定 1..300；其余开关只有布尔
  · 门禁在 auth.ADMIN_PREFIXES（仅管理员），nginx 与 FastAPI 两层都挡
"""
import os
import signal
import subprocess
import sys
import time

BILLI = '/collect/bili.py'
OUT_DIR = '/data-out'
LOG_DIR = '/logs'

MIN_COUNT, MAX_COUNT = 1, 300
_proc = None
_started = None
_log_path = None


def available():
    """能不能真的启停：脚本与日志目录都得挂进来。"""
    return os.path.exists(BILLI) and os.path.isdir(LOG_DIR)


def _running():
    global _proc
    if _proc is not None and _proc.poll() is not None:
        _proc = None                      # 自己结束了，清掉句柄
    return _proc is not None


def _tail(path, n=40):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return [l.rstrip('\n') for l in f.readlines()[-n:]]
    except OSError:
        return []


def start(count=10, skip_cooldown=False, fresh=True):
    """起一次采集。返回状态字典；已在跑或没挂进来则抛 RuntimeError。"""
    global _proc, _started, _log_path
    if not available():
        raise RuntimeError('采集脚本没有挂进容器（/collect/bili.py 不存在）—— '
                           '检查 docker-compose 里 fastapi 的 volumes。')
    if _running():
        raise RuntimeError('已经有一次采集在跑，等它结束或先停止。')
    count = int(count)
    if not (MIN_COUNT <= count <= MAX_COUNT):
        raise RuntimeError('每次采集数量需在 %d..%d 之间。' % (MIN_COUNT, MAX_COUNT))

    os.makedirs(LOG_DIR, exist_ok=True)
    _log_path = os.path.join(LOG_DIR, 'collect_web_%s.out' % time.strftime('%Y%m%d_%H%M%S'))
    # 参数以列表传递、不过 shell —— 网页永远改不了要跑什么命令
    cmd = [sys.executable, BILLI, 'hotlist', '-n', str(count)]
    if fresh:
        # --fresh：重新拉榜单、开新批次。不加就是复用本地榜单缓存
        # （批次身份绑榜单快照的 fetched_at 日期，所以不加会写进**旧批次**）。
        cmd.append('--fresh')
    if skip_cooldown:
        cmd.append('--no-init-cooldown')
    env = dict(os.environ)
    env['DANMAKU_DATA_DIR'] = OUT_DIR
    env['PYTHONUNBUFFERED'] = '1'
    logf = open(_log_path, 'w', encoding='utf-8')
    _proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                             env=env, cwd='/collect', start_new_session=True)
    _started = time.time()
    return status()


def stop():
    """SIGTERM 停：采集脚本自己装了 SIGTERM 处理（会落 manifest），
    所以中断是**可续采**的，不是硬杀。等 10 秒还不退才 kill。"""
    global _proc
    if not _running():
        return {'stopped': False, 'reason': '当前没有在跑的采集'}
    try:
        os.killpg(os.getpgid(_proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        _proc.terminate()
    for _ in range(20):
        if _proc.poll() is not None:
            break
        time.sleep(0.5)
    hard = _proc.poll() is None
    if hard:
        _proc.kill()
    _proc = None
    return {'stopped': True, 'hard_killed': hard}


def _newest_csv():
    try:
        names = [f for f in os.listdir(OUT_DIR) if f.endswith('.csv')]
    except OSError:
        return None, None
    if not names:
        return None, None
    full = [os.path.join(OUT_DIR, f) for f in names]
    newest = max(full, key=os.path.getmtime)
    return os.path.basename(newest), os.path.getmtime(newest)


def _snapshot_mtime():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3data', 'meta.json')
    return os.path.getmtime(p) if os.path.exists(p) else 0


def status():
    running = _running()
    newest, mtime = _newest_csv()
    st = {
        'available': available(),
        'running': running,
        'pid': (_proc.pid if running else None),
        'started_at': (time.strftime('%H:%M:%S', time.localtime(_started)) if _started else None),
        'elapsed_sec': (int(time.time() - _started) if (running and _started) else None),
        'log_path': _log_path,
        'log_tail': _tail(_log_path, 40),
        'newest_csv': newest,
        'newest_csv_at': (time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)) if mtime else None),
        'out_dir': OUT_DIR,
        # 采完还要在宿主机重导快照，前端才看得到新数据 —— 直接把这个信号给用户
        'snapshot_stale': bool(mtime and mtime > _snapshot_mtime()),
    }
    if not running and _proc is not None:
        st['exit_code'] = _proc.returncode
    return st
