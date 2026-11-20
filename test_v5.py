#!/usr/bin/env python3
"""memorykit v5 tests: per-turn upkeep, crash safety, missed sessions, upgrade.

    python3 test_v5.py                 # keyword mode (fast, stdlib only)
    MEMORYKIT_TEST_VECTOR=1 python3 test_v5.py   # also exercise the embedder

Every scenario runs the real CLI in a subprocess, so a "crash" is a real
os._exit(137) in the middle of a turn, not a simulated exception.
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MK = os.path.join(HERE, "memorykit.py")
LEGACY = os.environ.get("MEMORYKIT_LEGACY", "")  # path to a v4.5 memorykit.py
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def env(**kw):
    e = dict(os.environ)
    if os.environ.get("MEMORYKIT_TEST_VECTOR") != "1":
        e["MEMORYKIT_NO_EMBED"] = "1"
    e.update({k: str(v) for k, v in kw.items()})
    return e


def run(db, *args, stdin=None, script=MK, **kw):
    p = subprocess.run([sys.executable, script, "--db", db, *args], input=stdin,
                       capture_output=True, text=True, env=env(**kw))
    return p.returncode, p.stdout, p.stderr


def rows(db, where="1"):
    c = sqlite3.connect(db)
    out = c.execute(f"SELECT id, status, project, content FROM memory_item WHERE {where}").fetchall()
    c.close()
    return out


def logged_ids(db):
    d = os.path.join(os.path.dirname(db), "memory-log")
    ids = []
    if os.path.isdir(d):
        for f in os.listdir(d):
            for line in open(os.path.join(d, f), encoding="utf-8"):
                if "mk:{" in line:
                    rec = json.loads(line.split("<!-- mk:", 1)[1].rsplit(" -->", 1)[0])
                    if rec.get("op") == "capture":
                        ids.append(rec["id"])
    return ids


def turn(text, **kw):
    t = {"speaker": "user", "text": text, "session": "s1"}
    t.update(kw)
    return json.dumps(t) + "\n"


def fresh():
    d = tempfile.mkdtemp(prefix="mk5-")
    return os.path.join(d, "m.db")


def main():
    # 1. kill after the journal write, before the DB write
    db = fresh()
    rc, _, _ = run(db, "post-turn", "--jsonl", "--no-upkeep",
                   stdin=turn("we decided the cabin trip is in March, final"),
                   MEMORYKIT_TEST_CRASH_AT="after_journal")
    check("kill after journal: process really died", rc == 137, f"rc={rc}")
    n_before = len(rows(db))
    rc, out, _ = run(db, "pre-turn", "--query", "cabin trip")
    r = rows(db)
    check("kill after journal: next pre-turn stores the turn exactly once",
          n_before == 0 and len(r) == 1 and "cabin" in out, out.strip())
    check("kill after journal: logged exactly once", logged_ids(db).count(r[0][0]) == 1)

    # 2. kill after the DB commit, before the markdown log
    db = fresh()
    rc, _, _ = run(db, "post-turn", "--jsonl", "--no-upkeep",
                   stdin=turn("the dog's vet is Dr. Alvarez on Metcalf"),
                   MEMORYKIT_TEST_CRASH_AT="before_log")
    check("kill before log: row committed, log missing",
          rc == 137 and len(rows(db)) == 1 and logged_ids(db) == [])
    run(db, "post-turn", "--jsonl", stdin=turn("unrelated second turn about tea"))
    ids = logged_ids(db)
    check("kill before log: next call repairs the log, no duplicate row",
          len(rows(db)) == 2 and len(ids) == 2 and len(set(ids)) == 2, str(ids))
    rc, out, _ = run(db, "health")
    check("health clean after repair", rc == 0, out.strip()[:200])

    # 3. a torn half-written journal line is ignored
    db = fresh()
    run(db, "post-turn", "--jsonl", stdin=turn("first good turn, saved fine"))
    with open(db + ".journal.jsonl", "a") as fh:
        fh.write('{"jid": "abc", "speaker": "user", "te')
    rc, out, err = run(db, "pre-turn", "--query", "good turn")
    check("torn journal line tolerated", rc == 0 and len(rows(db)) == 1, err.strip()[:120])

    # 4. retrying the same turn (agent re-runs its hook) never duplicates
    db = fresh()
    t = turn("my sister's birthday is June 4", turn="t9")
    run(db, "post-turn", "--jsonl", stdin=t)
    run(db, "post-turn", "--jsonl", stdin=t)
    check("retried turn stored once", len(rows(db)) == 1)

    # 5. missed session: never ended, nobody calls sleep, a day passes
    db = fresh()
    old = "2026-01-01T00:00:00+00:00"
    texts = ["we decided: always back up the store before any migration",
             "the deadline for the garage quote is Friday, we agreed on it",
             "from now on never schedule anything on Sunday mornings"]
    batch = "".join(turn(x, session="never-closed", at=old, turn=str(i))
                    for i, x in enumerate(texts))
    run(db, "post-turn", "--jsonl", "--no-upkeep", stdin=batch)
    run(db, "post-turn", "--jsonl", stdin=turn("a new day, new session", session="s2"))
    sem = rows(db, "project='semantic'")
    check("missed session: aged turns distill on the next turn without end-session",
          len(sem) >= 2, f"semantic={len(sem)}")
    for _ in range(2):
        run(db, "post-turn", "--jsonl", stdin=turn(f"another turn {time.time()}", session="s2"))
    sem = rows(db, "project='semantic'")
    raw_live = rows(db, "project='turns' AND status!='rejected' AND session_key='never-closed'")
    check("missed session: backlog fully drained by bounded steps",
          len(sem) == 3 and not raw_live, f"semantic={len(sem)} raw_live={len(raw_live)}")

    # 6. GC is amortized: a stale low-salience row is quarantined by the
    #    rolling cursor within a few turns, with no gc command ever run
    db = fresh()
    stale = "".join(turn(f"ok sure thing number {i}", speaker="agent", source="a",
                         salience=1, at="2025-01-01T00:00:00+00:00", turn=f"z{i}")
                    for i in range(30))
    run(db, "post-turn", "--jsonl", "--no-upkeep", stdin=stale)
    calls = 0
    while calls < 6 and rows(db, "status!='rejected' AND content LIKE '%ok sure%'"):
        run(db, "post-turn", "--jsonl", stdin=turn(f"fresh turn {calls} about gardening"),
            MEMORYKIT_UPKEEP_GC_ROWS=10)
        calls += 1
    left = rows(db, "status!='rejected' AND content LIKE '%ok sure%'")
    check("gc amortized: stale rows quarantined over a few turns, bounded per call",
          not left and calls >= 3, f"calls={calls} left={len(left)}")
    c = sqlite3.connect(db)
    upk = json.loads(c.execute("SELECT value FROM memorykit_state WHERE key='upkeep_last'").fetchone()[0])
    mode = c.execute("PRAGMA journal_mode").fetchone()[0]
    c.close()
    check("upkeep records its last step; DB is in WAL mode", "at" in upk and mode == "wal",
          f"{upk} {mode}")

    # 7. dedup is amortized too
    db = fresh()
    dup = "".join(turn("Sam prefers dark roast coffee", speaker="user", turn=f"d{i}",
                       session=f"x{i}") for i in range(2))
    run(db, "post-turn", "--jsonl", stdin=dup)
    run(db, "post-turn", "--jsonl", stdin=turn("something else entirely here"))
    live = rows(db, "status!='rejected' AND content LIKE '%dark roast%'")
    check("dedup amortized: exact duplicate merged without a dedup command", len(live) == 1,
          f"live={len(live)}")

    # 8. pre-turn respects the token budget and the preferred-speaker lane
    db = fresh()
    many = "".join(turn(f"project falcon note {i}: " + "detail " * 40, speaker="agent",
                        source="bot", turn=f"f{i}") for i in range(10))
    many += turn("project falcon: I want it shipped by Friday", source="sam", turn="u1")
    run(db, "post-turn", "--jsonl", "--no-upkeep", stdin=many)
    rc, out, _ = run(db, "pre-turn", "--query", "project falcon", "--budget", "120",
                     "--prefer-source", "sam")
    approx = sum(len(l) // 4 + 1 for l in out.splitlines())
    check("pre-turn stays within budget", approx <= 120 + 70, f"~{approx} tokens")
    check("pre-turn puts the preferred speaker first", out.splitlines()[0].find("@sam") > 0,
          out.splitlines()[0][:80])

    # 9. provenance firewall still enforced through post-turn
    db = fresh()
    rc, _, err = run(db, "post-turn", "--speaker", "agent", "--provenance", "user-said",
                     "--text", "the agent tries to claim the user said this")
    check("firewall: agent cannot claim user-said via post-turn", rc == 2 and not os.path.exists(db) or not rows(db))

    # 10. upgrade an existing v4.5 database in place
    if LEGACY and os.path.exists(LEGACY):
        db = fresh()
        run(db, "init", script=LEGACY)
        for i in range(5):
            run(db, "capture", "--speaker", "user", "--text",
                f"legacy fact number {i} about the garden shed", script=LEGACY)
        before = rows(db)
        rc, out, _ = run(db, "upgrade")
        bak = [f for f in os.listdir(os.path.dirname(db)) if ".pre-v5-" in f]
        after = rows(db)
        check("upgrade: in place, backup made, rows untouched",
              rc == 0 and bak and before == after, out.strip())
        rc2, out2, _ = run(db, "upgrade")
        bak2 = [f for f in os.listdir(os.path.dirname(db)) if ".pre-v5-" in f]
        check("upgrade: idempotent (second run makes no second backup)",
              rc2 == 0 and len(bak2) == 1, out2.strip())
        rc, out, _ = run(db, "pre-turn", "--query", "garden shed")
        check("upgrade: old data searchable through the new hook", "garden shed" in out)
        run(db, "post-turn", "--jsonl", stdin=turn("first v5 turn after upgrade"))
        rc, out, _ = run(db, "health")
        check("upgrade: health clean afterwards", rc == 0, out.strip()[:200])
    else:
        print("SKIP  upgrade tests (set MEMORYKIT_LEGACY to a v4.5 memorykit.py)")

    # 10b. two processes replaying the same journal at once
    db = fresh()
    run(db, "init")
    recs = "".join(json.dumps({"jid": f"race{i:020d}", "speaker": "user", "session": "r",
                               "text": f"race turn number {i} about lanterns"}) + "\n"
                   for i in range(20))
    open(db + ".journal.jsonl", "w").write(recs)
    procs = [subprocess.Popen([sys.executable, MK, "--db", db, "pre-turn", "--query", "x"],
                              env=env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             for _ in range(3)]
    rcs = [p.wait() for p in procs]
    check("concurrent replay: no crash, every turn stored once",
          rcs == [0, 0, 0] and len(rows(db)) == 20, f"rcs={rcs} rows={len(rows(db))}")

    # 11. Python API
    sys.path.insert(0, HERE)
    os.environ.setdefault("MEMORYKIT_NO_EMBED", "1")
    import memorykit as mk
    db = fresh()
    ids = mk.post_turn(db, [{"speaker": "user", "text": "API test: the boat is named Wren"}])
    hits = mk.pre_turn(db, "boat name")
    check("python API: post_turn/pre_turn round trip", ids and "Wren" in hits, hits[:80])

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
