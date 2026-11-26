"""Distilled rows are logged by their distill record, so repair must not call
them db-only, and a log where 5.0.2 repair re-logged them as captures must
still reindex to the same rows."""
import os, subprocess, sys, tempfile, sqlite3, importlib.util
MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memorykit.py")
ENV = dict(os.environ, MEMORYKIT_NO_EMBED="1", MEMORYKIT_AUTO="0")
def run(db, *a):
    r = subprocess.run([sys.executable, MK, "--db", db, *a], capture_output=True, text=True, env=ENV)
    return r.returncode, r.stdout + r.stderr
def rows(db, where=""):
    c = sqlite3.connect(db); out = {r[0] for r in c.execute("select id from memory_item " + where)}; c.close(); return out
def test():
    t = tempfile.mkdtemp(); db = os.path.join(t, "x.db")
    run(db, "init")
    for i in range(8):
        run(db, "capture", "--speaker", "user", "--session", "s1", "--salience", "4",
            "--text", f"decision {i}: we ship the orchid build from rack {i} every friday")
    run(db, "end-session", "--session", "s1")
    rc, o = run(db, "sleep", "--apply", "--no-sweep")
    sem = rows(db, "where project='semantic'")
    if not sem:
        print("distill-repair: SKIP (sleep made no semantic row here)", o[-200:]); return
    rc, o = run(db, "repair")
    assert "0 db-only" in o, o
    # emulate 5.0.2: re-log a distilled row as a capture
    spec = importlib.util.spec_from_file_location("mk", MK); mk = importlib.util.module_from_spec(spec)
    os.environ.update(ENV); spec.loader.exec_module(mk)
    for i in sem:
        mk._log_from_row(db, i)
    before = rows(db)
    rc, o = run(db, "reindex", "--force")
    assert rc == 0 and "skipped" in o, o
    assert rows(db) == before, (len(rows(db)), len(before))
    print("distill-repair: ok")
if __name__ == "__main__":
    test()
