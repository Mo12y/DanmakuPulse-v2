#!/usr/bin/env python3
"""expD_count P0 gate: batch meta must exist, status=complete, windows non-empty.

Read-only. No model, no data loading beyond the two meta JSONs.
Exit 0 = pass; exit 2 = hard gate failure.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

METAS = {
    "20260902": "/home/molly/projects/danmaku/data/cli_collected/hotlist_top100_20260902.meta.json",
    "20260907": "/home/molly/projects/danmaku/data/cli_collected/hotlist_top200_20260907.meta.json",
}
OUT = "/home/molly/projects/danmaku-v2/experiments/expD_count/p0_evidence.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    result = {
        "check": "P0_batch_meta",
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "protocol": "experiments/expD_count/protocol.md",
        "rule": "meta exists AND status==complete AND len(windows)>0",
        "metas": {},
        "passed": True,
        "decision": None,
    }
    for batch, path in METAS.items():
        item = {"path": path, "exists": os.path.exists(path)}
        if item["exists"]:
            st = os.stat(path)
            item["size_bytes"] = st.st_size
            item["mtime"] = datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")
            item["sha256"] = sha256(path)
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            windows = d.get("windows") or []
            item.update({
                "batch_date": d.get("batch_date"),
                "csv": d.get("csv"),
                "list_fetched_at": d.get("list_fetched_at"),
                "status": d.get("status"),
                "attempted": d.get("attempted"),
                "ok": d.get("ok"),
                "failed": d.get("failed"),
                "coverage": d.get("coverage"),
                "n_windows": len(windows),
                "windows": windows,
                "note": d.get("note"),
            })
            item["status_complete"] = d.get("status") == "complete"
            item["windows_nonempty"] = len(windows) > 0
            item["pass"] = bool(item["status_complete"] and item["windows_nonempty"])
        else:
            item["pass"] = False
        result["metas"][batch] = item
        result["passed"] = result["passed"] and item["pass"]
    result["decision"] = "CONTINUE" if result["passed"] else "STOP_NO_FIT"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "passed": result["passed"],
        "decision": result["decision"],
        "per_batch": {b: {k: v.get(k) for k in ("exists", "status", "n_windows", "pass")}
                      for b, v in result["metas"].items()},
        "evidence": OUT,
    }, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
