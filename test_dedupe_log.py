"""5.0.4: `repair --dedupe-log` removes only provable copies, reindex names
every skipped id, and pre-turn repair runs only when asked and logs what it
touched."""
import glob, importlib.util, json, os, re, sqlite3, subprocess, sys, tempfile

MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memorykit.py")
ENV = dict(os.environ, MEMORYKIT_NO_EMBED="1", MEMORYKIT_AUTO="0", MEMORYKIT_SETTLE_HOURS="0")


def run(db, *a, env=None):
    r = subprocess.run([sys.executable, MK, "--db", db, *a], capture_output=True,
                       text=True, env=env or ENV)
    return r.returncode, r.stdout + r.stderr


def snapshot(db):
    c = sqlite3.connect(db)
    out = {r[0]: (r[1], r[2], r[3]) for r in c.execute(
        "select id, project, salience, content from memory_item")}
    c.close()
    return out


def log_ops(db):
    ops = {}
    for f in glob.glob(os.path.join(mod().md_dir_for(db), "*.md")):
        for line in open(f, encoding="utf-8"):
            m = re.search(r"<!-- mk:(\{.*\}) -->\s*$", line)
            if m:
                op = json.loads(m.group(1)).get("op")
                ops[op] = ops.get(op, 0) + 1
    return ops


def mod():
    os.environ.update(ENV)
    spec = importlib.util.spec_from_file_location("mk", MK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def store(n=10):
    t = tempfile.mkdtemp()
    db = os.path.join(t, "x.db")
    run(db, "init")
    for i in range(n):
        run(db, "capture", "--speaker", "user", "--session", "s1", "--salience", "5",
            "--text", f"fact {i}: the orchid build ships from rack {i} on friday")
    run(db, "end-session", "--session", "s1")
    run(db, "sleep", "--apply", "--no-sweep")
    return db


def semantic(db):
    c = sqlite3.connect(db)
    out = sorted(r[0] for r in c.execute("select id from memory_item where project='semantic'"))
    c.close()
    return out


def test_dedupe_damaged_and_clean():
    mk = mod()
    # clean store: nothing to remove
    db = store()
    sem = semantic(db)
    assert sem, "sleep distilled nothing"
    rc, o = run(db, "repair", "--dedupe-log")
    assert rc == 0 and "0 provable duplicate" in o, o
    # damage it the way 5.0.2 repair did
    for i in sem:
        mk._log_from_row(db, i)
    before_rows, before_ops = snapshot(db), log_ops(db)
    assert before_ops["capture"] > before_ops["distill"], before_ops
    rc, o = run(db, "reindex", "--force")
    assert rc == 0 and snapshot(db) == before_rows, o
    for i in sem:  # ids, not a count
        assert f"skipped duplicate capture {i} at " in o, (i, o)
    rc, o = run(db, "repair", "--dedupe-log")
    assert f"{len(sem)} provable duplicate" in o, o
    for i in sem:
        assert f"remove {i} at " in o, o
    assert log_ops(db) == before_ops, "dry run changed the log"
    rc, o = run(db, "repair", "--dedupe-log", "--apply")
    assert rc == 0 and f"removed {len(sem)}" in o and "backup:" in o, o
    for i in sem:
        assert f"removed {i} at " in o, o
    ops = log_ops(db)
    assert ops["capture"] == before_ops["capture"] - len(sem), (ops, before_ops)
    assert ops.get("dedupe_log") == 1, ops
    rc, o = run(db, "reindex", "--force")
    assert rc == 0 and "skipped" not in o, o
    assert snapshot(db) == before_rows
    rc, o = run(db, "repair", "--dedupe-log", "--apply")
    assert "nothing to remove" in o, o
    print("dedupe damaged+clean: ok")


def test_dedupe_refuses_unprovable():
    mk = mod()
    db = store(6)
    sem = semantic(db)
    a, b = sem[0], sem[1]
    mk._log_from_row(db, a)
    mk._log_from_row(db, b)
    # copy of a: row gone from the DB -> keep
    c = sqlite3.connect(db)
    c.execute("delete from memory_item where id=?", (a,))
    # copy of b: row text changed -> keep
    c.execute("update memory_item set content = content || ' (edited)' where id=?", (b,))
    c.commit()
    c.close()
    ops = log_ops(db)
    rc, o = run(db, "repair", "--dedupe-log", "--apply")
    assert f"keep   {a} at " in o and "row not in DB" in o, o
    assert f"keep   {b} at " in o and "differs" in o, o
    assert log_ops(db) == ops, "removed an unprovable record"
    print("dedupe refuses unprovable: ok")


def test_pre_turn_repair_gated_and_logged():
    db = store(4)
    c = sqlite3.connect(db)
    victim = sorted(r[0] for r in c.execute("select id from memory_item where project!='semantic'"))[-1]
    c.close()
    # make one db-only row: drop its log line
    for f in glob.glob(os.path.join(os.path.dirname(db), "**", "*.md"), recursive=True):
        lines = open(f, encoding="utf-8").readlines()
        keep = [ln for ln in lines if not (f'"id": "{victim}"' in ln and '"op": "capture"' in ln)]
        if keep != lines:
            open(f, "w", encoding="utf-8").writelines(keep)
    off = dict(ENV, MEMORYKIT_REPAIR_EVERY="0")
    for _ in range(3):
        rc, o = run(db, "pre-turn", "--query", "orchid", env=off)
    assert "repair" not in o, o
    assert log_ops(db).get("repair_run") is None
    every = dict(ENV, MEMORYKIT_REPAIR_EVERY="3")
    rc1, o1 = run(db, "pre-turn", "--query", "orchid", env=every)
    rc2, o2 = run(db, "pre-turn", "--query", "orchid", env=every)
    assert "repair" not in o1 + o2, o1 + o2
    rc3, o3 = run(db, "pre-turn", "--query", "orchid", env=every)
    assert f"pre-turn repair restored 0, logged 1: {victim}" in o3, o3
    assert log_ops(db).get("repair_run") == 1
    rc, o = run(db, "health")
    assert '"problems": []' in o, o
    # forced, on a clean store: nothing touched, nothing logged
    rc, o = run(db, "pre-turn", "--query", "orchid", "--repair", env=off)
    assert "repair" not in o and log_ops(db).get("repair_run") == 1, o
    print("pre-turn repair gated+logged: ok")


if __name__ == "__main__":
    test_dedupe_damaged_and_clean()
    test_dedupe_refuses_unprovable()
    test_pre_turn_repair_gated_and_logged()
