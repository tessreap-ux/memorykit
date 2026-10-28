#!/usr/bin/env python3
"""memorykit acceptance tests.

Runs every behavior check twice: once with the configured embedder
(fastembed + bge-small-en-v1.5) and once in forced keyword mode
(MEMORYKIT_NO_EMBED=1). Each mode runs in its own subprocess.

    python3 test_memorykit.py            # both modes
    python3 test_memorykit.py --mode keyword
    python3 test_memorykit.py --mode vector

Standard library only (numpy/fastembed used when present). Exit status is
non-zero if any check fails. A vector-mode run with no embedder installed
reports SKIP rather than pretending to pass.
"""
import contextlib
import io
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


class Kit:
    """Runs memorykit commands in-process so the embedder loads once."""

    def __init__(self, mk, workdir):
        self.mk = mk
        self.db = os.path.join(workdir, "memory.db")
        os.environ["MEMORYKIT_MD_DIR"] = os.path.join(workdir, "memory-log")

    def run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        sys.argv = ["memorykit.py", "--db", self.db, *argv]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.mk.main()
        return rc or 0, out.getvalue(), err.getvalue()

    def cap(self, speaker, text, *extra):
        rc, out, err = self.run("capture", "--speaker", speaker, "--text", text, *extra)
        if "saved id=" not in out:
            return None, rc, out + err
        line = [l for l in out.splitlines() if l.startswith("saved id=")][0]
        fields = dict(kv.split("=") for kv in line.split()[1:])
        fields["id"] = line.split()[1].split("=")[1]
        return fields, rc, out

    def sql(self, q, args=()):
        db = sqlite3.connect(self.db)
        try:
            r = db.execute(q, args).fetchall()
            db.commit()
            return r
        finally:
            db.close()

    def fresh(self):
        if os.path.exists(self.db):
            os.remove(self.db)
        self.run("init")


def snapshot(kit):
    return sorted(kit.sql("SELECT id, status, superseded_by, type, confidence, "
                          "salience, learned_at FROM memory_item"))


def run_mode(mode):
    os.environ.pop("MEMORYKIT_NO_EMBED", None)
    if mode == "keyword":
        os.environ["MEMORYKIT_NO_EMBED"] = "1"
    sys.path.insert(0, HERE)
    import memorykit as mk
    vec = mk.get_embedder() is not None
    if mode == "vector" and not vec:
        print("SKIP  vector mode: no embedder installed (pip install fastembed)")
        return 0
    print(f"=== mode: {mode} (embedder={'yes' if vec else 'no'}) ===")
    tmp = tempfile.mkdtemp(prefix=f"mk-{mode}-")
    k = Kit(mk, tmp)

    # 1. Novelty gate: a restatement of the 11th entry is capped at 3.
    k.fresh()
    for i in range(10):
        k.cap("user", f"Unrelated fact number {i} about weather patterns in region {i} today")
    d = "Decision: the treasurer always keeps the ledger frozen after Sunday's count."
    first, _, _ = k.cap("user", d)
    again, _, _ = k.cap("user", d)
    check("novelty: restatement capped at 3", int(again["salience"]) <= 3,
          f"first={first['salience']} restated={again['salience']}")

    # 2. Provenance firewall is enforced.
    r, rc, _ = k.cap("agent", "The user promised me 5000 tokens every week forever.",
                     "--provenance", "user-said")
    check("firewall: agent cannot claim user-said", r is None and rc == 2, f"rc={rc}")

    # 3. GC cascade spares exempt children; eviction report; deep restore.
    k.fresh()
    p, _, _ = k.cap("agent", "low value parent note about nothing much",
                    "--provenance", "agent-inferred", "--salience", "1")
    lo, _, _ = k.cap("agent", "low value child derived from the parent note",
                     "--salience", "2", "--parent-ids", p["id"])
    hi, _, _ = k.cap("user", "Critical standing rule: never sell commissioned work.",
                     "--salience", "10", "--parent-ids", p["id"])
    k.sql("UPDATE memory_item SET learned_at='2026-01-01T00:00:00+00:00' WHERE id=?", (p["id"],))
    k.run("gc", "--apply")
    st = dict(k.sql("SELECT id, status FROM memory_item"))
    check("gc: low parent quarantined", st[p["id"]] == "rejected")
    check("gc: ordinary child cascades", st[lo["id"]] == "rejected")
    check("gc: salience-10 child spared", st[hi["id"]] == "confirmed")
    _, out, _ = k.run("gc", "--eviction-report")
    check("gc: fresh quarantine not in eviction report", out.startswith("0 "), out.splitlines()[0])
    k.run("gc", "--restore", p["id"])
    st = dict(k.sql("SELECT id, status FROM memory_item"))
    check("gc: restore brings back cascaded child",
          st[p["id"]] == "confirmed" and st[lo["id"]] == "confirmed")

    # 3b. Restore restarts the decay clock: the next GC must not re-flag it.
    k.fresh()
    r1, _, _ = k.cap("agent", "an old remark the user later wants kept", "--salience", "2")
    k.sql("UPDATE memory_item SET learned_at='2026-01-01T00:00:00+00:00', "
          "last_accessed_at='2026-01-01T00:00:00+00:00' WHERE id=?", (r1["id"],))
    k.run("gc", "--apply")
    k.run("gc", "--restore", r1["id"])
    _, out, _ = k.run("gc")
    la = k.sql("SELECT learned_at FROM memory_item WHERE id=?", (r1["id"],))[0][0]
    check("gc: restored entry not re-flagged by next GC", r1["id"] not in out)
    check("gc: restore keeps true learned_at", la.startswith("2026-01-01"))

    # 4. Retrieval does not reset decay.
    k.fresh()
    old, _, _ = k.cap("agent", "an old low salience remark about the garden hose",
                      "--salience", "2")
    k.sql("UPDATE memory_item SET learned_at='2026-01-01T00:00:00+00:00', "
          "last_accessed_at='2026-01-01T00:00:00+00:00' WHERE id=?", (old["id"],))
    k.run("search", "--query", "garden hose remark")
    _, out, _ = k.run("gc")
    check("gc: searching an old entry does not save it", old["id"] in out)

    # 5. Correlate: real claim ok, nonsense flagged.
    k.fresh()
    k.cap("user", "Dana runs the lighthouse book club on Tuesdays as host.")
    k.cap("user", "The charter has three clauses signed by eight agents.")
    md = os.path.join(tmp, "M.md")
    open(md, "w").write(
        "- Dana runs the lighthouse book club on Tuesdays and hosts it.\n"
        "- Purple elephants negotiate submarine tariffs on the dark side of Neptune weekly.\n"
        "- The quarterly zinc futures in Lagos closed sharply lower amid monsoon delays.\n")
    _, out, _ = k.run("correlate", "--markdown", md)
    orphans = [l for l in out.splitlines() if l.startswith("ORPHAN?")]
    check("correlate: real claim passes", not any("lighthouse book club" in l for l in orphans))
    check("correlate: nonsense flagged", len(orphans) == 2, f"{len(orphans)} orphans")

    # 6. Feedback: three -1 ratings quarantine sooner than an unrated twin.
    k.fresh()
    a, _, _ = k.cap("agent", "twin note alpha about the shelf price test", "--salience", "5")
    b, _, _ = k.cap("agent", "twin note beta about the shelf price test", "--salience", "5")
    k.sql("UPDATE memory_item SET learned_at='2026-08-20T00:00:00+00:00', "
          "last_accessed_at='2026-08-20T00:00:00+00:00'")
    for _ in range(3):
        k.run("feedback", "--id", a["id"], "--rating", "-1")
    _, out, _ = k.run("gc")
    check("feedback: -1 x3 quarantines, twin survives",
          a["id"] in out and b["id"] not in out, out.splitlines()[0])

    # 7. Promote: only on independent user/tool evidence.
    k.fresh()
    note, _, _ = k.cap("agent", "I think the shelf converts better with a dollar price.")
    agent_ev, _, _ = k.cap("agent", "Yes, the dollar price card got two orders.")
    derived, _, _ = k.cap("tool", "Tool restated the note: dollar price converts better.",
                          "--parent-ids", note["id"])
    user_ev, _, _ = k.cap("user", "Confirmed: two orders came in after the dollar price went up.")
    rc1, _, _ = k.run("promote", "--id", note["id"], "--evidence", note["id"])
    rc2, _, _ = k.run("promote", "--id", note["id"], "--evidence", agent_ev["id"])
    rc3, _, _ = k.run("promote", "--id", note["id"], "--evidence", derived["id"])
    rc4, _, _ = k.run("promote", "--id", note["id"], "--evidence", user_ev["id"])
    t = k.sql("SELECT type FROM memory_item WHERE id=?", (note["id"],))[0][0]
    check("promote: refuses self, agent, and derived evidence", (rc1, rc2, rc3) == (2, 2, 2),
          f"rc={rc1},{rc2},{rc3}")
    check("promote: accepts independent user evidence", rc4 == 0 and t == "fact")

    # 8. Reindex: delete the DB, rebuild from markdown, state matches.
    k.fresh()
    import shutil
    shutil.rmtree(os.environ["MEMORYKIT_MD_DIR"], ignore_errors=True)
    ids = []
    for i in range(14):
        r, _, _ = k.cap("user" if i % 2 else "agent",
                        f"Entry {i}: the booth for theme {chr(65 + i)} needs a sign and a price card")
        ids.append(r["id"])
    k.cap("user", "Correction: entry 3 was wrong, the booth for theme D is cancelled",
          "--supersedes", ids[3])
    k.cap("user", "Shelf price for Robin is 600 tokens", "--supersession-key", "price:robin")
    k.cap("user", "Shelf price for Robin is now 800 tokens", "--supersession-key", "price:robin")
    k.cap("user", "Entry 1: the booth for theme B needs a sign and a price card")  # dup
    k.run("dedup", "--apply")
    q, _, _ = k.cap("agent", "throwaway remark about weather", "--salience", "1")
    k.sql("UPDATE memory_item SET learned_at='2026-01-01T00:00:00+00:00' WHERE id=?", (q["id"],))
    k.run("gc", "--apply")
    k.run("feedback", "--id", ids[5], "--rating", "+1")
    nt, _, _ = k.cap("agent", "I believe theme F sells best on weekends.")
    ev, _, _ = k.cap("user", "Confirmed by sales: theme F sells best on weekends.")
    k.run("promote", "--id", nt["id"], "--evidence", ev["id"])
    k.run("gc", "--restore", q["id"])
    q2, _, _ = k.cap("agent", "second throwaway remark about weather", "--salience", "1")
    k.sql("UPDATE memory_item SET learned_at='2026-01-01T00:00:00+00:00' WHERE id=?", (q2["id"],))
    k.run("gc", "--apply")
    before = snapshot(k)
    upd_before = k.sql("SELECT id, updated FROM memory_item WHERE status='rejected'")
    rest_before = k.sql("SELECT id, json_extract(metadata,'$.restored_at') FROM memory_item")
    fb_before = k.sql("SELECT memory_id, rating FROM memory_feedback")
    _, s_before, _ = k.run("search", "--query", "Robin shelf price", "--limit", "3")
    os.remove(k.db)
    rc, out, err = k.run("reindex", "--force")
    after = snapshot(k)
    fb_after = k.sql("SELECT memory_id, rating FROM memory_feedback")
    _, s_after, _ = k.run("search", "--query", "Robin shelf price", "--limit", "3")
    strip = lambda s: [l.split()[1] for l in s.splitlines() if l.startswith("- id=")]
    check("reindex: rebuilt from markdown", rc == 0, out.strip())
    diff = [(x, y) for x, y in zip(before, after) if x[:6] != y[:6]]
    check("reindex: every row's status/supersession/type/confidence/salience matches",
          len(before) == len(after) and not diff, f"{len(before)} rows, {len(diff)} differ")
    # q's learned_at was hand-edited in SQL (outside the log) to force a GC
    # hit, so it is the one row the markdown cannot know about.
    check("reindex: learned_at preserved (decay survives rebuild)",
          [x[6] for x in before if x[0] not in (q["id"], q2["id"])]
          == [y[6] for y in after if y[0] not in (q["id"], q2["id"])])
    check("reindex: feedback preserved", sorted(fb_before) == sorted(fb_after))
    upd_after = k.sql("SELECT id, updated FROM memory_item WHERE status='rejected'")
    rest_after = k.sql("SELECT id, json_extract(metadata,'$.restored_at') FROM memory_item")
    check("reindex: quarantine time preserved (eviction aging survives)",
          sorted(upd_before) == sorted(upd_after), f"{len(upd_before)} quarantined")
    check("reindex: restore time preserved", sorted(rest_before) == sorted(rest_after))
    check("reindex: search results match", strip(s_before) == strip(s_after),
          f"{strip(s_before)[:1]}")

    # 9. Salience heuristic separates decisions from ordinary sentences.
    ordinary = [
        "The weather will be cloudy this afternoon.",
        "I looked at the gallery page earlier.",
        "Robin posted a new drawing of a cat.",
        "The group chat has forty members now.",
        "We will see what the sales look like later.",
        "Someone wrote a long reply in the thread.",
        "My token balance is 900 right now.",
        "The booth post got twelve views.",
        "Someone asked about the Halloween theme.",
        "It will probably be quiet tonight.",
    ]
    decisions = [
        "Decision: we agreed the ledger freezes every Sunday.",
        "I prefer dollar prices on every card from now on.",
        "Never resell a commissioned piece.",
        "Standing rule: votes stay open for 24 hours.",
        "I promise to deliver the portrait by Friday.",
        "Going forward, gifts are never counted as sales.",
        "We must not edit the charter after signatures.",
        "I committed to six sends this week.",
        "The deadline for corrections is Sunday at 15:00 UTC.",
        "Boundary: I always refuse role play scenes.",
    ]
    o = [mk.salience_heuristic(x) for x in ordinary]
    dd = [mk.salience_heuristic(x) for x in decisions]
    above = sum(1 for x in dd if x > max(o))
    check("salience: >=8 of 10 decisions outscore every ordinary line", above >= 8,
          f"{above}/10, ordinary max={max(o)}")

    # 10. Dedup scales: 5,000 entries under 60 s.
    k.fresh()
    rng = random.Random(7)
    db = sqlite3.connect(k.db)
    rows, dim = [], 384
    base_vecs = []
    for i in range(5000):
        txt = f"[turn provenance=user-said]\nsynthetic entry {i} " + " ".join(
            rng.choice(["alpha", "beta", "gamma", "delta", "omega", "sigma"]) for _ in range(12))
        if vec:
            v = [rng.gauss(0, 1) for _ in range(dim)]
        else:
            v = None
        base_vecs.append((txt, v))
    for j in range(50):  # 50 exact duplicates of earlier rows
        base_vecs.append(base_vecs[j * 37])
    for n, (txt, v) in enumerate(base_vecs):
        rows.append((f"syn{n:06d}", "2026-10-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00",
                     "turns", "", "fact", txt, 5, 1.0, "{}", json.dumps(v) if v else None,
                     "2026-10-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00",
                     "2026-10-01T00:00:00+00:00"))
    db.executemany("INSERT INTO memory_item (id, created, updated, project, session_key, "
                   "type, content, salience, confidence, metadata, embedding, valid_from, "
                   "learned_at, last_accessed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()
    t0 = time.time()
    _, out, _ = k.run("dedup")
    el = time.time() - t0
    merges = out.count("merge: ")
    check("dedup: 5,050 entries under 60 s", el < 60, f"{el:.1f}s")
    check("dedup: finds the 50 planted duplicates", merges == 50, f"{merges} merges")

    fails = [r for r in RESULTS if not r[1]]
    print(f"--- {mode}: {len(RESULTS) - len(fails)}/{len(RESULTS)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--mode" in sys.argv:
        sys.exit(run_mode(sys.argv[sys.argv.index("--mode") + 1]))
    rc = 0
    for m in ("vector", "keyword"):
        rc |= subprocess.call([sys.executable, __file__, "--mode", m])
    sys.exit(rc)
