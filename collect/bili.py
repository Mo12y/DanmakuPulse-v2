#!/usr/bin/env python3
"""
B站弹幕采集CLI工具 — 终端独立运行，不需要AI对话

用法:
  python bili.py hotlist                    # 采集热门榜 Top 100 弹幕（默认）
  python bili.py hotlist -n 50              # 采集热门榜 Top 50
  python bili.py hotlist --fresh            # 抓新榜 → 新批次快照（不覆盖已存在的批次快照）
  python bili.py hotlist -n 50 --no-fresh --batch-date 20260914   # 断点续采同一批

  python bili.py video BVxxx                # 采集单个视频弹幕
  python bili.py video BV1xx BV2xx BV3xx    # 采集多个视频弹幕
  python bili.py video -f bvids.txt         # 从文件读取BV号列表

  python bili.py search "游戏" -n 30         # 搜索关键词，采集搜索结果弹幕
  python bili.py search "游戏" --order pubdate --tids 4  # 按最新发布排序，限定游戏分区

采集策略（基于实测限流参数）:
  批次内: 每10个视频一批，视频间间隔10秒
  批次间: 冷却12分钟（避开XML 412封堵，实测验证）
  断点续采: 中断后重新运行自动跳过已完成的视频

批次身份（2026-09-17 起，见 docs/01-设计/采集批次身份与断点续采_任务书_v1.md 批次 A）:
  - 批次日 = 榜单快照的 fetched_at 日期（不是"开始跑的那天"）→ CSV 名与批次清单都不漂
  - 榜单快照按批次落盘 data/cli_collected/hotlist_videos_<batch_date>.json，--fresh 绝不覆盖
  - 批次清单 <csv>.meta.json 记录 windows/coverage/status（部分批次必须可区分）
  - 视频级采集时刻落盘 data/cli_collected/fetch_times_<batch_date>.csv（bvid,fetched_at,src；
    2026-10-14 起）——M4 纵向 delta 按采集时刻记账（batch_date 只是日期标签，1012 批
    实际跨 26 小时三段窗口）
  - 采集顺序**全局随机置换** → 中断留下的是随机子集，不是榜单前缀；
    名次请查榜单快照：python scripts/rank_from_snapshot.py --batch-date <batch_date>

环境变量:
  DANMAKU_DATA_DIR  覆盖输出目录（测试/验收用，避免测试 CSV 落进正式 data/cli_collected/）

依赖: pip install requests
"""

import sys
import os
import time
import csv
import random
import re
import json
import argparse
import requests
from datetime import datetime
from collections import Counter

# ── 全局配置 ─────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}

# 采集参数（均经过实测验证）
BATCH_SIZE = 10             # 每批视频数——XML API在~15次请求后触发412
REQUEST_DELAY = 10.0         # 视频间延迟（秒）——实测安全的间隔
COOLDOWN_MINUTES = 12        # 批次间冷却（分钟）——2026-08-13隔离测试验证12min安全(4批0拦截)，原15min为保守值
INITIAL_COOLDOWN = 5         # 首次启动冷却（分钟）——清空之前可能积累的限流

# 默认输出目录（项目根 = collect/ 的上一级）
# DANMAKU_DATA_DIR 可覆盖输出目录——仅供测试/验收把产物写进临时目录，
# 避免测试用 CSV 落进 data/cli_collected/ 被 clean.py 的 glob 当成正式批次。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DANMAKU_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "cli_collected"))
os.makedirs(DATA_DIR, exist_ok=True)

# ── 批次身份（见 docs/01-设计/采集批次身份与断点续采_任务书_v1.md 批次 A）──
# 批次身份绑「榜单快照的 fetched_at 日期」，不绑「开始跑的那天」。
LEGACY_VLIST_FILE = "hotlist_videos.json"          # 旧的无批次缓存（兼容读取）


# ── 跨平台终端安全输出 ──────────────────────────────────────
# Windows 控制台默认 GBK，需降级替换；Linux/macOS 为 UTF-8，直接输出。

IS_WINDOWS = sys.platform == "win32"

def safe_print(*args, **kwargs):
    """跨平台安全打印：Windows 做 GBK 降级，Linux/macOS 直接 UTF-8 输出"""
    if not IS_WINDOWS:
        print(*args, **kwargs)
        return
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for a in args:
            if isinstance(a, str):
                safe_args.append(
                    a.encode('gbk', errors='replace').decode('gbk', errors='replace')
                )
            else:
                safe_args.append(a)
        print(*safe_args, **kwargs)


# ── API 层 ───────────────────────────────────────────────────

def api_get(sess, url, params=None, timeout=20):
    """GET JSON 响应，含重试和412检测"""
    for attempt in range(3):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json(), None
            elif r.status_code == 412:
                return None, "RATE_LIMITED"
            else:
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                return None, str(e)
    return None, "MAX_RETRIES"


def fetch_popular(sess, max_pages=10):
    """获取热门榜视频列表（翻到底或达到max_pages）"""
    safe_print("\n  [热门榜] 获取中...")
    videos = []
    seen = set()

    for pn in range(1, max_pages + 1):
        data, err = api_get(sess, "https://api.bilibili.com/x/web-interface/popular",
                           params={"ps": 50, "pn": pn})
        if err:
            safe_print(f"    pn={pn}: API错误 {err}")
            break

        d = data.get("data", {})
        vlist = d.get("list", [])
        no_more = d.get("no_more", False)

        added = 0
        for v in vlist:
            bvid = v.get("bvid", "")
            if bvid and bvid not in seen:
                seen.add(bvid)
                stat = v.get("stat", {})
                owner = v.get("owner", {})
                videos.append({
                    "bvid": bvid,
                    "cid": v.get("cid", 0),
                    "videos": v.get("videos", 1),   # 分P数（多P视频的 cid 只是 P1）
                    "title": v.get("title", ""),
                    "duration": v.get("duration", 0),
                    "tname": v.get("tname", ""),
                    "play": stat.get("view", 0),
                    "danmaku_reported": stat.get("danmaku", 0),
                    "like": stat.get("like", 0),
                    "coin": stat.get("coin", 0),
                    "favorite": stat.get("favorite", 0),
                    "share": stat.get("share", 0),
                    "reply": stat.get("reply", 0),
                    "pubdate": v.get("pubdate", 0),
                    "up_mid": str(owner.get("mid", "")),
                    "up_name": owner.get("name", ""),
                    "source": "popular",
                })
                added += 1

        safe_print(f"    pn={pn}: +{added} (累计 {len(videos)}), no_more={no_more}")

        if no_more or added == 0:
            break
        time.sleep(0.3)

    safe_print(f"  热门榜合计: {len(videos)} 个视频")
    return videos


def fetch_ranking(sess, rid=0, max_requests=1):
    """获取分区排行榜视频列表（ranking/v2，用于对照组采集）

    参数:
      rid: 分区ID（0=全站。游戏=4, 生活=160, 知识=36, 音乐=3, 美食=211, 影视=181）
      max_requests: 请求次数（ranking/v2 单次固定返回~100条，无翻页）
    """
    safe_print(f"\n  [分区榜] rid={rid} 获取中...")
    videos = []
    seen = set()

    for i in range(max_requests):
        data, err = api_get(sess, "https://api.bilibili.com/x/web-interface/ranking/v2",
                            params={"rid": rid, "type": "all"})
        if err:
            safe_print(f"    req{i+1}: API错误 {err}")
            break
        if not data or data.get("code") != 0:
            safe_print(f"    req{i+1}: code={data.get('code') if data else 'None'} {data.get('message','') if data else ''}")
            break

        vlist = data.get("data", {}).get("list", [])
        if not vlist:
            safe_print(f"    req{i+1}: 空列表")
            break

        added = 0
        for v in vlist:
            bvid = v.get("bvid", "")
            if bvid and bvid not in seen:
                seen.add(bvid)
                stat = v.get("stat", {})
                owner = v.get("owner", {})
                videos.append({
                    "bvid": bvid,
                    "cid": v.get("cid", 0),
                    "videos": v.get("videos", 1),   # 分P数（多P视频的 cid 只是 P1）
                    "title": v.get("title", ""),
                    "duration": v.get("duration", 0),
                    "tname": v.get("tname", ""),
                    "play": stat.get("view", 0),
                    "danmaku_reported": stat.get("danmaku", 0),
                    "like": stat.get("like", 0),
                    "coin": stat.get("coin", 0),
                    "favorite": stat.get("favorite", 0),
                    "share": stat.get("share", 0),
                    "reply": stat.get("reply", 0),
                    "pubdate": v.get("pubdate", 0),
                    "up_mid": str(owner.get("mid", "")),
                    "up_name": owner.get("name", ""),
                    "source": f"ranking:{rid}",
                })
                added += 1
        safe_print(f"    req{i+1}: +{added} (累计 {len(videos)})")

    safe_print(f"  分区榜合计: {len(videos)} 个视频")
    return videos


def fetch_search(sess, keyword, max_results=100, order="pubdate", tids=0):
    """搜索视频并按条件返回"""
    safe_print(f"\n  [搜索] '{keyword}' order={order} tids={tids}...")
    videos = []
    seen = set()

    for pn in range(1, 20):
        if len(videos) >= max_results:
            break

        params = {
            "search_type": "video",
            "keyword": keyword,
            "order": order,
            "page": pn,
        }
        if tids:
            params["tids"] = tids

        data, err = api_get(sess, "https://api.bilibili.com/x/web-interface/search/type",
                           params=params)
        if err:
            safe_print(f"    pn={pn}: API错误 {err}")
            break

        results = data.get("data", {}).get("result", [])
        if not results:
            break

        for v in results:
            bvid = v.get("bvid", "")
            if bvid and bvid not in seen:
                seen.add(bvid)
                videos.append({
                    "bvid": bvid,
                    "cid": 0,  # search API 不返回 cid
                    "title": v.get("title", ""),
                    "duration": v.get("duration", "0"),
                    "tname": "",  # search API 可能不返回分区名
                    "play": v.get("play", 0),
                    "danmaku_reported": v.get("video_review", 0),
                    "up_mid": str(v.get("mid", "")),
                    "up_name": v.get("author", ""),
                    "source": f"search:{keyword}",
                })

        safe_print(f"    pn={pn}: 累计 {len(videos)}")

        if len(results) < 20:
            break
        time.sleep(0.5)

    safe_print(f"  搜索合计: {len(videos)} 个视频 (目标 {max_results})")
    return videos[:max_results]


def resolve_cid(sess, bvid):
    """通过 view API 查询视频的 cid（含分P列表 pages）"""
    data, err = api_get(sess, "https://api.bilibili.com/x/web-interface/view",
                       params={"bvid": bvid})
    if err or not data or data.get("code") != 0:
        return None
    d = data.get("data", {})
    stat = d.get("stat", {})
    pages = d.get("pages") or []
    return {
        "cid": d.get("cid", 0),
        "duration": d.get("duration", 0),
        "title": d.get("title", ""),
        "play": stat.get("view", 0),
        "danmaku_reported": stat.get("danmaku", 0),
        "like": stat.get("like", 0),
        "coin": stat.get("coin", 0),
        "favorite": stat.get("favorite", 0),
        "share": stat.get("share", 0),
        "reply": stat.get("reply", 0),
        "pubdate": d.get("pubdate", 0),
        "tname": d.get("tname", ""),
        "up_mid": str(d.get("owner", {}).get("mid", "")),
        "up_name": d.get("owner", {}).get("name", ""),
        # 分P：d['pages'] = [{cid, page, part, duration}, ...]；权威判据（统计判据会漏）
        "videos": d.get("videos", len(pages) or 1),
        "pages": [{"cid": p.get("cid"), "page": p.get("page"),
                   "part": p.get("part", ""), "duration": p.get("duration", 0)}
                  for p in pages],
    }


def video_parts(v):
    """归一化分P列表。多P 用 view API 的 pages；否则退化成一个 P（单P路径不变）。"""
    pages = v.get("pages") or []
    if len(pages) > 1:
        return pages
    return [{"cid": v.get("cid", 0), "page": 1, "part": "",
             "duration": v.get("duration", 0)}]


def fetch_danmaku_all_parts(sess, v, inter_part_delay=0):
    """逐 P 拉弹幕，并把 time_offset 按前序各 P 时长累加拼到**整片时间轴**（方案 A）。

    返回 (danmakus, status)，status 语义与 fetch_danmaku_xml 一致。
    单P视频只走一个 P，且直接透传原状态码 → 行为与改动前完全一致（零回归）。
    """
    parts = video_parts(v)
    if len(parts) == 1:
        return fetch_danmaku_xml(sess, parts[0].get("cid") or 0)

    all_dms, offset = [], 0.0
    for i, p in enumerate(parts):
        cid = p.get("cid") or 0
        if cid:
            dms, status = fetch_danmaku_xml(sess, cid)
            if status == "blocked":
                return all_dms, "blocked"
            # 只在 P2 及以后做偏移；第一个 P 的 offset=0.0，x+0.0 与 x 逐位相同
            for d in (dms or []):
                d["time_offset"] = d["time_offset"] + offset
            all_dms.extend(dms or [])
        offset += float(p.get("duration") or 0)
        if inter_part_delay and i < len(parts) - 1:
            time.sleep(inter_part_delay)
    return all_dms, ("ok" if all_dms else "empty_pool")


def fetch_danmaku_xml(sess, cid):
    """XML弹幕——返回 (弹幕列表, 状态)"""
    try:
        r = sess.get("https://api.bilibili.com/x/v1/dm/list.so",
                     params={"oid": cid}, timeout=20)
        if r.status_code == 412:
            return None, "blocked"
        if r.status_code != 200:
            return [], f"http_{r.status_code}"
        r.encoding = "utf-8"
        dms = []
        for attr, text in re.findall(r'<d p="([^"]+)"[^>]*>(.*?)</d>', r.text):
            p = attr.split(",")
            if len(p) >= 8:
                dms.append({
                    "time_offset": float(p[0]),
                    "mode": int(p[1]),
                    "font_size": int(p[2]),
                    "color": p[3],
                    "send_time": int(p[4]),
                    "pool": int(p[5]),
                    "user_hash": p[6],
                    "row_id": int(p[7]),
                    "text": text.strip(),
                    "text_len": len(text.strip()),
                })
        status = "ok" if dms else "empty_pool"
        return dms, status
    except Exception as e:
        return [], f"err:{str(e)[:40]}"


# ── 采集核心逻辑 ─────────────────────────────────────────────

CSV_COLS = [
    "bvid", "cid", "up_mid", "up_name", "title", "duration", "play",
    "danmaku_reported", "danmaku_got",
    "like", "coin", "favorite", "share", "reply", "pubdate",
    "time_offset", "send_time", "mode", "font_size", "color",
    "text", "text_len", "pool", "row_id", "user_hash",
    "status", "source", "tname",
    "part_count", "part_cids",   # 分P：单P视频留空；多P=各P cid 逗号分隔
]
# user_hash（2026-09-28 补）：弹幕发送者的**哈希 ID**，来自弹幕 XML `p` 属性第 7 段。
#   它**一直被 `fetch_danmaku` 解析**（见本文件 `"user_hash": p[6]`），却**从未写进 CSV** ——
#   典型的**静默丢字段**：无报错、无警告、无人发现，直到 2026-09-28 做 R5 可行性检验才暴露。
#   影响：已有 4 批（529 视频 / 67.7 万弹幕）**没有 uid** ⇒ 观众结构分析只能重采。
#   新批次会自动带该列；**旧 CSV 无此列**（下游 `usecols` 读取不受影响）。
#   它也是**跨视频二部图**的唯一依据，但前提是"同一观众跨请求哈希稳定"—— 该稳定性**尚未验证**。


def load_completed_bvids(csv_path):
    """从已有CSV读取已完成的bvid（支持断点续采）"""
    if not os.path.exists(csv_path):
        return set()
    completed = set()
    try:
        with open(csv_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("status") in ("ok", "empty_pool"):
                    completed.add(row.get("bvid", ""))
    except Exception:
        pass
    return completed


def save_batch_csv(results, csv_path, write_header=False):
    """追加写入CSV。

    ⚠️ **列定义与写入行的唯一来源是 `CSV_COLS`**（2026-09-28 重构）。
    旧写法列名写一处、取值表达式硬编码另一处（29 个），**加字段时漏改任一处就会
    静默写出错位的列**（`csv.writer` 不校验长度：短行补空、长行照写）。
    ⇒ 现按 `CSV_COLS` 逐列取值，写入前断言所有列都在 row 里，**把静默错位变成显式失败**。
    """
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_COLS)
        DANMAKU_KEYS = ("time_offset", "send_time", "mode", "font_size", "color",
                        "text", "text_len", "pool", "row_id", "user_hash")
        for r in results:
            dms = r.get("danmakus", []) or []
            base = {
                "bvid": r["bvid"], "cid": r.get("cid", 0),
                "up_mid": r.get("up_mid", ""), "up_name": r.get("up_name", ""),
                "title": r.get("title", ""), "duration": r.get("duration", 0),
                "play": r.get("play", 0),
                "danmaku_reported": r.get("danmaku_reported", 0),
                "like": r.get("like", 0), "coin": r.get("coin", 0),
                "favorite": r.get("favorite", 0), "share": r.get("share", 0),
                "reply": r.get("reply", 0), "pubdate": r.get("pubdate", 0),
                "source": r.get("source", ""), "tname": r.get("tname", ""),
                "part_count": r.get("part_count", ""), "part_cids": r.get("part_cids", ""),
            }
            for d in (dms if dms else [None]):
                row = dict(base)
                row["danmaku_got"] = r.get("danmaku_got", 0) if dms else 0
                row["status"] = r.get("status", "") if dms else r.get("status", "no_data")
                for k in DANMAKU_KEYS:
                    # ⚠️ 取不到写空串（失败视频本无弹幕字段）；**不静默用 0 顶替**
                    row[k] = d.get(k, "") if d else ""
                missing = [c for c in CSV_COLS if c not in row]
                assert not missing, 'save_batch_csv 缺列 %s —— 列定义与取值不同步' % missing
                w.writerow([row[c] for c in CSV_COLS])


# ── 批次身份：榜单快照 per-batch 落盘 + 批次清单 sidecar ──────

def batch_cache_path(batch_date):
    """批次化榜单快照路径：hotlist_videos_<batch_date>.json"""
    return os.path.join(DATA_DIR, f"hotlist_videos_{batch_date}.json")


def save_batch_cache(batch_date, videos, fetched_at):
    """写本批次的榜单快照（原子写）。绝不覆盖已存在的同批次快照——调用方必须先检查。"""
    path = batch_cache_path(batch_date)
    payload = {"fetched_at": fetched_at, "count": len(videos), "videos": videos}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def load_vlist_file(path):
    """读一个榜单快照文件 → (videos, fetched_at)；不存在返回 (None, None)。"""
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("videos", []), data.get("fetched_at")


def date_of(iso_ts):
    """'2026-09-14T13:12:42.944222' → '20260914'；解析不了返回 None。"""
    if not iso_ts:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(iso_ts))
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None


class BatchManifest:
    """批次清单 sidecar：与 CSV 同名，后缀换 .meta.json。

    字段对齐 data/cli_collected/hotlist_top200_20260914.meta.json（该文件的 schema 即本任务书 A3）。
    每次启动追加一个 window；正常收尾 status=complete，被 SIGTERM/异常收尾 status=aborted。
    这是"两段式快照"可见化的唯一手段——部分批次与完整批次必须可区分。

    口径（2026-09-17 修 N2）：顶层 attempted/ok/failed/coverage 是**整批累计**，
    windows[].videos_added 才是**本轮窗口**的量。断点续采时 __init__ 加载已有清单并**保留累计**，
    只由 start_window() 追加新窗口——不再把整批数字重置成本轮窗口（否则 9/17 那种"显示 45/100、
    整批实为 100/100"的错值会重演）。
    """

    def __init__(self, csv_path, batch_date, list_fetched_at, list_count, count):
        self.path = (csv_path[:-4] if csv_path.endswith(".csv") else csv_path) + ".meta.json"
        if os.path.exists(self.path):
            # 断点续采：加载已有清单，保留整批累计的 attempted/ok/failed/windows/started_at
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.d = json.load(f)
            except Exception:
                self.d = None
            if not isinstance(self.d, dict) or "windows" not in self.d:
                self.d = self._fresh(batch_date, csv_path, list_fetched_at, list_count, count)
                self._save()
        else:
            self.d = self._fresh(batch_date, csv_path, list_fetched_at, list_count, count)
            self._save()
        self._load_times()

    @staticmethod
    def _fresh(batch_date, csv_path, list_fetched_at, list_count, count):
        return {
            "batch_date": batch_date,
            "csv": os.path.basename(csv_path),
            "list_fetched_at": list_fetched_at,
            "list_count": list_count,
            "count": count,
            "status": "collecting",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "windows": [],
            "attempted": 0,
            "ok": 0,
            "failed": 0,
            "coverage": f"0/{count}",
        }

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ── 视频级采集时刻 sidecar（2026-10-14 新增；M4 纵向「按采集时刻记账」输入）──
    # 落盘 <DATA_DIR>/fetch_times_<batch_date>.csv（列：bvid,fetched_at,src）。
    # 纪律：写入失败只告警、绝不中断采集（采集完整性优先于 sidecar）。
    def _times_path(self):
        bd = self.d.get("batch_date") or "unknown"
        return os.path.join(os.path.dirname(self.path), f"fetch_times_{bd}.csv")

    def _load_times(self):
        self._times = {}
        p = self._times_path()
        try:
            if os.path.exists(p):
                with open(p, encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        if row.get("bvid"):
                            self._times[row["bvid"]] = (row.get("fetched_at", ""),
                                                        row.get("src", "collector"))
        except Exception as e:
            safe_print(f"  [WARN] fetch_times sidecar 读取失败（忽略，继续采集）：{e}")

    def note_time(self, bvid, ts=None):
        """记录一个视频的采集时刻（_save_one 调用；任何异常不外抛）。"""
        try:
            self._times[bvid] = (ts or datetime.now().isoformat(timespec="seconds"),
                                 "collector")
            self._write_times()
        except Exception as e:
            safe_print(f"  [WARN] fetch_times sidecar 写入失败（忽略，继续采集）：{e}")

    def _write_times(self):
        if not self._times:
            return
        p = self._times_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bvid", "fetched_at", "src"])
            for bvid in sorted(self._times):
                t, src = self._times[bvid]
                w.writerow([bvid, t, src])
        os.replace(tmp, p)

    def start_window(self):
        self.d["windows"].append({
            "start": datetime.now().isoformat(timespec="seconds"),
            "end": None,
            "videos_added": 0,
        })
        self.d["status"] = "collecting"
        self._save()

    def note_video(self, status):
        """每写一个视频调用一次：更新当前 window 与**整批累计**计数。"""
        if self.d["windows"]:
            self.d["windows"][-1]["videos_added"] += 1
        self.d["attempted"] += 1
        if status in ("ok", "empty_pool"):
            self.d["ok"] += 1
        else:
            self.d["failed"] += 1
        self.d["coverage"] = f"{self.d['ok']}/{self.d['count']}"

    def finish(self, status):
        """收尾：关闭当前 window（若开着）、写 status、落盘。"""
        if self.d["windows"] and self.d["windows"][-1].get("end") is None:
            self.d["windows"][-1]["end"] = datetime.now().isoformat(timespec="seconds")
        self.d["status"] = status
        self.d["ended_at"] = datetime.now().isoformat(timespec="seconds")
        self._save()
        return self.path


def _install_sigterm(manifest):
    """SIGTERM → 收尾写 manifest(status=aborted) → 退出。

    修 F6：/api/collect/stop 发的是 SIGTERM，python 无 handler 会立即死，
    日志尾部与批次清单一起丢。这里保证"掐断"也能留下可读的批次身份。
    """
    def handler(signum, frame):
        safe_print("\n  [SIGTERM] 收到终止信号，正在收尾（写批次清单 status=aborted）...")
        try:
            if manifest:
                p = manifest.finish("aborted")
                d = manifest.d
                safe_print(f"  已采统计: ok={d.get('ok')} / attempted={d.get('attempted')} / "
                           f"failed={d.get('failed')} / coverage={d.get('coverage')}")
                safe_print(f"  批次清单已更新: {p}")
        except Exception as e:
            safe_print(f"  [WARN] 批次清单收尾失败: {e}")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(143)

    try:
        import signal
        signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass  # 平台不支持时降级为原行为（不阻断采集）


def batch_crawl(videos, csv_path, label="",
                batch_size=BATCH_SIZE, request_delay=REQUEST_DELAY,
                cooldown_minutes=COOLDOWN_MINUTES, randomize=None,
                manifest=None, shuffle=True, rng_seed=None):
    """
    分批次采集弹幕——核心采集循环

    策略: 每批 batch_size 个视频 → 批次间冷却 cooldown_minutes 分钟
    支持: 断点续采、412自动冷却重试、进度显示、视频级保存

    randomize: None=固定参数 | "delay"=DELAY±20%+COOLDOWN±20%随机 | "batch"=BATCH±20%随机

    shuffle : 采集前对 pending 做一次**全局随机置换**（2026-09-17 起默认开）。
              目的：中断时留下的是"Top N 的随机子集"，而不是"Top N 的前缀"——
              名次与 8 维节奏实测 0/8 显著（审计 v2 §一），前缀偏差没有数据上的正当理由，
              而它正是 9/07 与 9/14 两次事故的同一类病灶。
              名次不再由 CSV 首次出现顺序推断，改为查本批榜单快照（scripts/rank_from_snapshot.py）。
    rng_seed: 仅供测试复现随机顺序；None = 系统熵。
    """
    completed = load_completed_bvids(csv_path)
    pending = [v for v in videos if v["bvid"] not in completed]

    if shuffle:
        (random.Random(rng_seed) if rng_seed is not None else random).shuffle(pending)
    total = len(pending)

    if not pending:
        safe_print("  [OK] 所有视频已完成采集")
        return

    total_batches = (total + batch_size - 1) // batch_size
    cycle_time = (batch_size * request_delay + cooldown_minutes * 60) / 60
    est_hours = total_batches * cycle_time / 60
    rnd_label = {"delay": " (DELAY/COOLDOWN随机)", "batch": " (BATCH随机)"}.get(randomize, "")

    safe_print(f"\n{'='*60}")
    safe_print(f"  {label}采集任务")
    safe_print(f"  待采集: {total} 视频 | 已完成: {len(completed)} 视频")
    safe_print(f"  批次大小: {batch_size}{rnd_label} | 视频间隔: {request_delay}s | 批次冷却: {cooldown_minutes}min")
    safe_print(f"  采集顺序: {'全局随机置换（名次请查榜单快照）' if shuffle else '按榜单名次顺序'}")
    safe_print(f"  预计 {total_batches} 批次, 每批 ~{cycle_time:.0f}min, 总计 ~{est_hours:.1f}h")
    safe_print(f"  输出: {csv_path}")
    safe_print(f"{'='*60}")

    sess = requests.Session()
    sess.headers.update(HEADERS)

    total_dm = 0
    stats = {"ok": 0, "empty": 0, "blocked": 0, "no_cid": 0}
    batch_num = (len(completed) // batch_size) + 1
    first_write = (len(completed) == 0)

    def _save_one(v):
        """立即保存单个视频——视频级断点续采（中断最多丢1个视频）"""
        nonlocal first_write
        save_batch_csv([v.copy()], csv_path, write_header=first_write)
        first_write = False
        if manifest is not None:
            manifest.note_video(v.get("status", ""))
            manifest.note_time(v["bvid"])   # 视频级采集时刻（fetch_times sidecar）
            manifest._save()

    idx = 0
    while idx < total:
        # 每批参数随机化(假设A: delay随机; 假设B: batch随机)
        if randomize == "batch":
            bs = random.randint(8, 12)
        else:
            bs = batch_size
        if randomize == "delay":
            d = random.uniform(8.0, 12.0)
            cd = random.uniform(12.0, 18.0)
        else:
            d = request_delay
            cd = cooldown_minutes

        batch = pending[idx:idx + bs]
        timestamp = datetime.now().strftime("%H:%M:%S")

        safe_print(f"\n  -- 批次 {batch_num}/{total_batches} [{timestamp}] "
                   f"({idx+1}-{idx+len(batch)}/{total}) --")

        ok_count = 0
        empty_count = 0
        blocked = False

        for i, v in enumerate(batch):
            bvid = v["bvid"]

            # 如果视频没有 cid，先查询
            if v.get("cid", 0) == 0:
                safe_print(f"    [{idx+i+1}/{total}] {bvid} 查询cid...", end=" ", flush=True)
                info = resolve_cid(sess, v)
                if info:
                    v.update(info)
                    safe_print(f"cid={v['cid']}", end=" ", flush=True)
                else:
                    v["status"] = "view_api_failed"
                    _save_one(v)
                    stats["no_cid"] += 1
                    safe_print("FAIL")
                    continue
                time.sleep(1)

            cid = v.get("cid", 0)
            rpt = v.get("danmaku_reported", 0)

            if cid == 0:
                v["status"] = "no_cid"
                _save_one(v)
                stats["no_cid"] += 1
                continue

            # 分P：popular/分区榜只给 P1 的 cid；多P视频必须再查一次 view API 拿全 parts
            # （统计判据会漏掉"P1 占比>80%"的多P视频，view API 的 pages 才是权威）
            if (v.get("videos") or 1) > 1 and not v.get("pages"):
                safe_print(f"    [{idx+i+1}/{total}] {bvid} 多P({v.get('videos')}) 查询各P...",
                           end=" ", flush=True)
                info = resolve_cid(sess, bvid)
                if info and info.get("pages"):
                    v["pages"] = info["pages"]
                    if info.get("duration"):
                        v["duration"] = info["duration"]   # view API 的总时长为准
                    safe_print(f"{len(info['pages'])} 个P", end=" ", flush=True)
                else:
                    safe_print("view API 失败→退回只采 P1", end=" ", flush=True)
                time.sleep(1)

            parts = video_parts(v)
            if len(parts) > 1:
                v["part_count"] = len(parts)
                v["part_cids"] = ",".join(str(p.get("cid") or 0) for p in parts)

            dms, status = fetch_danmaku_all_parts(
                sess, v, inter_part_delay=(d if len(parts) > 1 else 0))

            if status == "blocked":
                v["danmaku_got"] = 0
                v["danmakus"] = []
                v["status"] = "blocked"
                _save_one(v)
                stats["blocked"] += 1
                blocked = True
                safe_print(f"    [{idx+i+1}/{total}] {bvid} [BLOCKED-412] 批次中断，进入冷却")
                break

            got = len(dms) if dms else 0
            v["danmaku_got"] = got
            v["danmakus"] = dms if dms else []
            v["status"] = status
            _save_one(v)

            total_dm += got
            if dms:
                ok_count += 1
            else:
                empty_count += 1

            # 显示进度（标题做GBK安全处理，仅 Windows 需要）
            title = (v.get("title") or "")[:35]
            title_safe = title.encode('gbk', errors='replace').decode('gbk', errors='replace') if IS_WINDOWS else title
            rate = got / rpt * 100 if rpt else 0
            trunc = " [截断!]" if (rpt > 3000 and rate < 50) else ""
            safe_print(f"    [{idx+i+1}/{total}] {bvid} rpt={rpt} "
                       f"got={got} ({rate:.0f}%){trunc} {title_safe}")

            time.sleep(d)

        stats["ok"] += ok_count
        stats["empty"] += empty_count

        safe_print(f"    批次 {batch_num}: OK={ok_count} 空={empty_count} "
                   f"累计OK={stats['ok']} 累计弹幕={total_dm:,}")

        idx += bs
        batch_num += 1

        # 批次间冷却
        remaining = total - idx
        if remaining > 0 or blocked:
            cooldown_sec = cd * 60
            resume_time = datetime.fromtimestamp(datetime.now().timestamp() + cooldown_sec)
            safe_print(f"\n  [冷却 {cd:.1f}min] 下次启动预计 {resume_time.strftime('%H:%M:%S')}")
            safe_print(f"  进度: {idx}/{total} ({idx/total*100:.0f}%)  "
                       f"| 可Ctrl+C中断，重新运行自动续采")
            try:
                time.sleep(cooldown_sec)
            except KeyboardInterrupt:
                safe_print(f"\n\n  [中断] 已采集数据已保存。重新运行 bili.py 即可续采。")
                safe_print(f"  下次将从第 {idx+1} 个视频继续（共 {total} 个）")
                return

    # 完成
    safe_print(f"\n{'='*60}")
    safe_print(f"  [完成] {label}")
    safe_print(f"  OK: {stats['ok']} | 空池: {stats['empty']} | "
               f"拦截: {stats['blocked']} | 无cid: {stats['no_cid']}")
    safe_print(f"  总弹幕: {total_dm:,} 条")
    safe_print(f"  输出: {csv_path}")
    safe_print(f"{'='*60}")


# ── 命令实现 ─────────────────────────────────────────────────

def _resolve_hotlist_list(args, sess):
    """Phase 1：确定本批的榜单快照 + batch_date（批次身份绑 fetched_at 日期）。

    返回 (videos, fetched_at, batch_date, list_count, cache_path)。
    任何"该用哪份榜单说不清"的情形一律报错退出，**禁止静默改抓新榜**（修 F2）。
    """
    explicit = getattr(args, "batch_date", None)
    if explicit and not re.fullmatch(r"\d{8}", explicit):
        safe_print(f"  [FAIL] --batch-date 必须是 YYYYMMDD（收到 {explicit!r}）")
        sys.exit(2)

    legacy_path = os.path.join(DATA_DIR, LEGACY_VLIST_FILE)

    if args.fresh:
        # ── 抓新榜 → 写本批次快照（绝不覆盖已存在的快照）──
        videos = fetch_popular(sess, max_pages=args.pages)
        if not videos:
            safe_print("  [FAIL] 未获取到任何视频")
            sys.exit(1)
        fetched_at = datetime.now().isoformat()
        batch_date = explicit or date_of(fetched_at)
        cache_path = batch_cache_path(batch_date)
        if os.path.exists(cache_path):
            safe_print(f"\n  [FAIL] 该批次的榜单快照已存在，--fresh 拒绝覆盖它：")
            safe_print(f"         {cache_path}")
            safe_print(f"         榜单快照一旦被覆盖，'上一批基于哪份榜单'就永久丢失（F2）。")
            safe_print(f"         要用它做断点续采：去掉 --fresh，改用 "
                       f"--no-fresh --batch-date {batch_date}")
            safe_print(f"         要开新批次：换一个 --batch-date（或跨天后再采）")
            sys.exit(3)
        save_batch_cache(batch_date, videos, fetched_at)
        safe_print(f"\n  榜单快照已保存: {cache_path}")
        return videos, fetched_at, batch_date, len(videos), cache_path

    # ── 复用缓存（断点续采）──
    if explicit:
        batch_date = explicit
        cache_path = batch_cache_path(batch_date)
        videos, fetched_at = load_vlist_file(cache_path)
        if videos is None:
            # 兼容：9/15 之前的批次只有旧的无批次缓存；仅当它的 fetched_at 正是该批次日才认
            lv, lf = load_vlist_file(legacy_path)
            if lv and date_of(lf) == batch_date:
                safe_print(f"  [兼容] 未找到 {os.path.basename(cache_path)}，"
                           f"回退到旧缓存 {LEGACY_VLIST_FILE}（fetched_at={lf} 与该批次日一致）")
                videos, fetched_at, cache_path = lv, lf, legacy_path
            else:
                safe_print(f"\n  [FAIL] --no-fresh 找不到批次 {batch_date} 的榜单快照：")
                safe_print(f"         {cache_path}")
                safe_print(f"         拒绝静默改抓新榜（否则本批身份与实际榜单不符，"
                           f"正是 8/12 事故的成因）。")
                sys.exit(4)
    else:
        # 未指定 batch_date 且未 --fresh：只能用旧的无批次缓存
        videos, fetched_at = load_vlist_file(legacy_path)
        if videos is None:
            safe_print(f"\n  [FAIL] --no-fresh 但既没有批次化快照、也没有 {LEGACY_VLIST_FILE}")
            sys.exit(4)
        batch_date = date_of(fetched_at)
        if batch_date is None:
            safe_print(f"  [FAIL] 缓存里的 fetched_at 无法解析：{fetched_at!r}")
            sys.exit(4)
        cache_path = legacy_path
        safe_print(f"  [兼容] 使用旧缓存 {LEGACY_VLIST_FILE} → 推定批次日 {batch_date}")

    safe_print(f"\n  复用榜单快照: {cache_path}")
    safe_print(f"  获取时间: {fetched_at} | 共 {len(videos)} 个视频 | 批次日: {batch_date}")
    return videos, fetched_at, batch_date, len(videos), cache_path


def cmd_hotlist(args):
    """采集热门榜弹幕"""
    safe_print("=" * 60)
    safe_print(f"  B站热门榜弹幕采集 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 60)

    sess = requests.Session()
    sess.headers.update(HEADERS)

    # Phase 1: 榜单快照（批次身份 = 快照 fetched_at 日期，不是"今天"）
    videos, fetched_at, batch_date, list_count, _cache = _resolve_hotlist_list(args, sess)

    # 取 Top N
    target = videos[:args.count]
    safe_print(f"  目标: Popular Top {args.count}")

    # Phase 2: 弹幕采集
    prefix = args.prefix or f"hotlist_top{args.count}"
    csv_path = os.path.join(DATA_DIR, f"{prefix}_{batch_date}.csv")
    manifest = BatchManifest(csv_path, batch_date, fetched_at, list_count, args.count)
    manifest.start_window()
    _install_sigterm(manifest)
    safe_print(f"  批次清单: {manifest.path}")

    # 初始冷却（清空可能积累的限流；--no-init-cooldown 用于测试跳过）
    _initial_cooldown(args, manifest)

    status = "complete"
    try:
        batch_crawl(target, csv_path, label=f"热门榜Top{args.count}",
                    batch_size=args.batch,
                    cooldown_minutes=args.cooldown,
                    randomize=None if args.randomize == "none" else args.randomize,
                    manifest=manifest,
                    shuffle=not getattr(args, "no_shuffle", False))
    except KeyboardInterrupt:
        status = "aborted"
        safe_print(f"\n\n  [中断] 已采集数据已保存到: {csv_path}")
        safe_print(f"  续采（同一批、同一份榜单）："
                   f"python collect/bili.py hotlist -n {args.count} --no-fresh --batch-date {batch_date}")
    finally:
        p = manifest.finish(status)
        safe_print(f"  批次清单已更新: {p}  (status={status}, coverage={manifest.d['coverage']})")


def _initial_cooldown(args, manifest=None):
    """初始冷却。Ctrl-C = 退出采集（不再"跳过冷却直接开采"——A4）。"""
    if args.no_init_cooldown:
        safe_print("  [跳过初始冷却] --no-init-cooldown 指定")
        return True
    safe_print(f"\n  初始冷却 {INITIAL_COOLDOWN}min（清空限流状态）...")
    safe_print("  （Ctrl+C = 退出本次采集，不会开始采集）")
    try:
        for t in range(INITIAL_COOLDOWN * 60, 0, -30):
            time.sleep(30)
            if t % 300 == 0:
                safe_print(f"    剩余 ~{t//60}min...")
    except KeyboardInterrupt:
        safe_print("\n  [退出] 冷却期间被中断——本次不采集，未写入任何数据。")
        if manifest is not None:
            manifest.finish("aborted")
        sys.exit(130)
    return True


def cmd_ranklist(args):
    """采集分区排行榜弹幕（对照组用）"""
    safe_print("=" * 60)
    safe_print(f"  B站分区排行榜弹幕采集 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"  rid={args.rid} (对照组分区), Top {args.count}")
    safe_print("=" * 60)

    sess = requests.Session()
    sess.headers.update(HEADERS)

    # Phase 1: 获取分区榜
    videos = fetch_ranking(sess, rid=args.rid)
    if not videos:
        safe_print("  [FAIL] 未获取到任何视频")
        return

    # 取 Top N
    target = videos[:args.count]
    safe_print(f"  目标: rid={args.rid} Top {args.count}")

    # Phase 2: 弹幕采集（批次身份同样绑 fetched_at 日期；分区榜不做缓存复用）
    fetched_at = datetime.now().isoformat()
    explicit = getattr(args, "batch_date", None)
    if explicit and not re.fullmatch(r"\d{8}", explicit):
        safe_print(f"  [FAIL] --batch-date 必须是 YYYYMMDD（收到 {explicit!r}）")
        sys.exit(2)
    batch_date = explicit or date_of(fetched_at)
    prefix = args.prefix or f"rank_rid{args.rid}"
    csv_path = os.path.join(DATA_DIR, f"{prefix}_{batch_date}.csv")
    manifest = BatchManifest(csv_path, batch_date, fetched_at, len(videos), args.count)
    manifest.start_window()
    _install_sigterm(manifest)
    safe_print(f"  批次清单: {manifest.path}")

    # 初始冷却（可跳过）
    _initial_cooldown(args, manifest)

    status = "complete"
    try:
        batch_crawl(target, csv_path, label=f"分区榜rid{args.rid}",
                    batch_size=args.batch,
                    cooldown_minutes=args.cooldown,
                    randomize=None if args.randomize == "none" else args.randomize,
                    manifest=manifest,
                    shuffle=not getattr(args, "no_shuffle", False))
    except KeyboardInterrupt:
        status = "aborted"
        safe_print(f"\n\n  [中断] 已采集数据已保存到: {csv_path}")
        safe_print(f"  重新运行 'python bili.py ranklist --rid {args.rid}' 即可续采")
    finally:
        p = manifest.finish(status)
        safe_print(f"  批次清单已更新: {p}  (status={status}, coverage={manifest.d['coverage']})")


def cmd_video(args):
    """采集指定BV号的弹幕"""
    bvids = args.bvids

    # 如果从文件读取
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            file_bvids = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        bvids = (bvids or []) + file_bvids

    if not bvids:
        safe_print("错误: 请指定至少一个BV号，或使用 -f 从文件读取")
        sys.exit(1)

    # 去重
    bvids = list(dict.fromkeys(bvids))

    safe_print("=" * 60)
    safe_print(f"  单视频弹幕采集 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"  共 {len(bvids)} 个视频")
    safe_print("=" * 60)

    sess = requests.Session()
    sess.headers.update(HEADERS)

    # 先查询所有视频的cid
    safe_print("\n  [查询视频信息]")
    videos = []
    for i, bvid in enumerate(bvids):
        safe_print(f"    [{i+1}/{len(bvids)}] {bvid} ...", end=" ", flush=True)
        info = resolve_cid(sess, bvid)
        if info:
            info["bvid"] = bvid
            info["source"] = "manual"
            videos.append(info)
            safe_print(f"cid={info['cid']} | {info['danmaku_reported']}弹幕 | "
                       f"{info['title'][:30]}")
        else:
            safe_print("FAIL (视频不存在或API错误)")
            videos.append({"bvid": bvid, "cid": 0, "source": "manual",
                           "title": "", "duration": 0, "play": 0,
                           "danmaku_reported": 0})
        time.sleep(1)

    csv_path = os.path.join(DATA_DIR, f"videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # 不做批次冷却——少量视频直接采
    batch_crawl(videos, csv_path, label="单视频")


def cmd_search(args):
    """搜索并采集弹幕"""
    safe_print("=" * 60)
    safe_print(f"  搜索采集 — '{args.keyword}' — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 60)

    sess = requests.Session()
    sess.headers.update(HEADERS)

    # Phase 1: 搜索
    videos = fetch_search(sess, args.keyword, max_results=args.count,
                          order=args.order, tids=args.tids)

    if not videos:
        safe_print("  [FAIL] 搜索结果为空")
        return

    # Phase 2: 弹幕采集
    prefix = args.prefix or args.keyword.replace(" ", "_")
    csv_path = os.path.join(DATA_DIR, f"search_{prefix}_{datetime.now().strftime('%Y%m%d')}.csv")

    try:
        batch_crawl(videos, csv_path, label=f"搜索'{args.keyword}'")
    except KeyboardInterrupt:
        safe_print(f"\n\n  [中断] 已采集数据已保存到: {csv_path}")


# ── CLI 入口 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="B站弹幕采集CLI工具 — 终端独立运行，用于演示和数据采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python bili.py hotlist                     采集热门榜 Top 100
  python bili.py hotlist -n 50               采集热门榜 Top 50
  python bili.py hotlist --fresh              强制刷新榜单（不使用缓存）

  python bili.py video BV1xx4y1B7Eu          采集单个视频
  python bili.py video -f bvids.txt           从文件批量采集

  python bili.py search "游戏实况" -n 30      搜索并采集30个视频
  python bili.py search "影评" --order click  按播放量排序搜索

采集参数（实测验证）:
  批次=10个视频, 间隔=10s, 批次冷却=12min
  支持 Ctrl+C 中断后重新运行自动续采 (断点续采)
        """
    )

    sub = parser.add_subparsers(dest="command", help="子命令")

    # ── hotlist 子命令 ──
    p_hot = sub.add_parser("hotlist", help="采集热门榜弹幕",
        description="从B站热门榜(popular API)获取视频，分批次采集弹幕")
    p_hot.add_argument("-n", "--count", type=int, default=100,
                       help="采集视频数量 (默认: 100)")
    p_hot.add_argument("--pages", type=int, default=10,
                       help="热门榜翻页数 (默认: 10)")
    p_hot.add_argument("--fresh", action="store_true",
                       help="强制重新获取榜单（不使用本地缓存）")
    p_hot.add_argument("--batch", type=int, default=BATCH_SIZE,
                       help=f"每批视频数 (默认: {BATCH_SIZE})")
    p_hot.add_argument("--cooldown", type=int, default=COOLDOWN_MINUTES,
                       help=f"批次冷却分钟 (默认: {COOLDOWN_MINUTES})")
    p_hot.add_argument("--randomize", choices=["none", "delay", "batch"],
                       default="none",
                       help="参数随机化: none(固定)/delay(DELAY+COOLDOWN±20%%的抖动)/batch(BATCH±20%%的抖动)")
    p_hot.add_argument("--prefix", default=None,
                       help="输出文件名前缀 (默认: hotlist_top{count})")
    p_hot.add_argument("--no-init-cooldown", action="store_true",
                       help="跳过初始冷却（测试用）")
    p_hot.add_argument("--batch-date", default=None, metavar="YYYYMMDD",
                       help="批次日期（默认=榜单快照的 fetched_at 日期）。决定 CSV 名与批次清单，"
                            "跑过午夜/隔日续采都不会漂。")
    p_hot.add_argument("--no-shuffle", action="store_true",
                       help="关闭采集顺序的全局随机置换（默认开启；仅调试用）")

    # ── ranklist 子命令 ──
    p_rank = sub.add_parser("ranklist", help="采集分区排行榜弹幕（对照组用）",
        description="从B站分区排行榜(ranking/v2)获取视频，分批次采集弹幕。用于对照组：分区头部视频 vs 全站热榜")
    p_rank.add_argument("--rid", type=int, default=4,
                        help="分区ID (默认: 4=游戏。生活=160, 知识=36, 音乐=3, 美食=211, 影视=181)")
    p_rank.add_argument("-n", "--count", type=int, default=100,
                        help="采集视频数量 (默认: 100)")
    p_rank.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"每批视频数 (默认: {BATCH_SIZE})")
    p_rank.add_argument("--cooldown", type=int, default=COOLDOWN_MINUTES,
                        help=f"批次冷却分钟 (默认: {COOLDOWN_MINUTES})")
    p_rank.add_argument("--randomize", choices=["none", "delay", "batch"],
                        default="none",
                        help="参数随机化: none(固定)/delay/delay+batch")
    p_rank.add_argument("--prefix", default=None,
                        help="输出文件名前缀 (默认: rank_rid{rid})")
    p_rank.add_argument("--no-init-cooldown", action="store_true",
                        help="跳过初始冷却（测试用）")
    p_rank.add_argument("--batch-date", default=None, metavar="YYYYMMDD",
                        help="批次日期（默认=抓榜当天）")
    p_rank.add_argument("--no-shuffle", action="store_true",
                        help="关闭采集顺序的全局随机置换（默认开启；仅调试用）")

    # ── video 子命令 ──
    p_vid = sub.add_parser("video", help="采集指定BV号弹幕",
        description="采集一个或多个BV号的弹幕，支持从文件读取")
    p_vid.add_argument("bvids", nargs="*", default=[],
                       help="BV号列表")
    p_vid.add_argument("-f", "--file",
                       help="从文件读取BV号（一行一个，#开头为注释）")

    # ── search 子命令 ──
    p_src = sub.add_parser("search", help="搜索视频并采集弹幕",
        description="按关键词搜索B站视频，采集搜索结果弹幕")
    p_src.add_argument("keyword", help="搜索关键词")
    p_src.add_argument("-n", "--count", type=int, default=50,
                       help="采集视频数量 (默认: 50)")
    p_src.add_argument("--order", default="pubdate",
                       choices=["pubdate", "click", "dm", "stow", "scores"],
                       help="排序方式: pubdate(最新)/click(播放多)/dm(弹幕多) (默认: pubdate)")
    p_src.add_argument("--tids", type=int, default=0,
                       help="分区ID (0=全部。游戏=4, 生活=36, 知识=96, 影视=188)")
    p_src.add_argument("--prefix", help="输出文件名前缀 (默认: 关键词)")

    args = parser.parse_args()

    if args.command == "hotlist":
        cmd_hotlist(args)
    elif args.command == "ranklist":
        cmd_ranklist(args)
    elif args.command == "video":
        cmd_video(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
