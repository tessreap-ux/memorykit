#!/usr/bin/env python3
"""Chat-log adapter for memorykit v5: one process per turn.

stdin : JSON lines exported from a chat log ({id, role, time, text}),
        possibly empty.
argv  : --db PATH  [--query Q] [--budget N] [--user-source NAME]
stdout: "stored N" (only when turns came in), then the recall lines.

Replaces ingest.py + recall.py (which spawned one `memorykit capture`
process, and one model load, per turn). Turn ids are deterministic, so
re-sending a batch after a dropped connection never duplicates anything.
"""
import argparse, json, os, sys
_here = os.path.dirname(os.path.abspath(__file__))
for _d in (os.environ.get("MEMORYKIT_LIB", ""), os.path.dirname(os.path.dirname(_here)),
           os.path.dirname(_here), "/opt/memorykit/lib"):
    if _d and os.path.exists(os.path.join(_d, "memorykit.py")):
        sys.path.insert(0, _d)
        break
import memorykit as mk

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--query", default="")
ap.add_argument("--budget", type=int, default=250)
ap.add_argument("--user-source", default="user")
ap.add_argument("--agent-source", default="agent")
a = ap.parse_args()


def own_words(t):
    """The user's own lines, without quoted or pasted blocks."""
    keep = [l for l in t.split("\n") if not l.lstrip().startswith(">") and not l.startswith("  ")]
    return "\n".join(keep).strip()


turns = []
if not sys.stdin.isatty():
    for line in sys.stdin:
        if not line.strip():
            continue
        d = json.loads(line)
        user = d["role"] == "user"
        stamp = f"[{d.get('time', '')}] " if d.get("time") else ""
        base = {"session": "chat:" + d.get("time", "")[:10], "turn": d["id"][:8],
                "speaker": "user" if user else "agent",
                "provenance": "user-said" if user else "agent-said",
                "source": a.user_source if user else a.agent_source}
        t = d["text"]
        if user:
            w = own_words(t)
            if w and w != t.strip() and len(w) >= 8:
                turns.append({**base, "turn": base["turn"] + ":own", "text": stamp + w})
        turns.append({**base, "text": stamp + t})
if turns:
    ids = mk.post_turn(a.db, turns)
    print(f"stored {len(ids)}")
if a.query.strip():
    print(mk.pre_turn(a.db, a.query, a.budget, prefer_source=a.user_source) or "(no matches)")
