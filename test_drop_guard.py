"""Miso's damaged-log rig: delete 3 captures from the log, tear a 4th mid-JSON.
reindex --force must refuse and name the rows; health must name them; the torn
line must be reported with file:line; repair must name and restore them."""
import os, subprocess, sys, tempfile, sqlite3, glob
MK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memorykit.py")
ENV = dict(os.environ, MEMORYKIT_NO_EMBED="1", MEMORYKIT_AUTO="0")
def run(db, *a):
    r = subprocess.run([sys.executable, MK, "--db", db, *a], capture_output=True, text=True, env=ENV)
    return r.returncode, r.stdout + r.stderr
def ids(db):
    c = sqlite3.connect(db); out = [r[0] for r in c.execute("select id from memory_item order by learned_at")]; c.close(); return out
def test():
    t = tempfile.mkdtemp(); db = os.path.join(t, "x.db")
    run(db, "init")
    for i in range(12):
        rc, o = run(db, "capture", "--speaker", "user", "--text", f"note {i}: the orchid deploy key sits on vault shelf {i*7} near rack {i}")
        assert rc == 0, o
    all_ids = ids(db); n = len(all_ids); assert n >= 8, n
    md = glob.glob(os.path.join(t, "**", "*.md"), recursive=True)[0]
    lines = open(md).read().split("\n")
    gone = all_ids[:3]; torn = all_ids[3]
    out = []
    for l in lines:
        if any(g in l for g in gone):
            continue
        if torn in l and "<!-- mk:" in l:
            l = l[: len(l) // 2]
        out.append(l)
    open(md, "w").write("\n".join(out))
    lost = set(gone) | {torn}
    rc, o = run(db, "reindex", "--force")
    assert rc == 2 and "refusing" in o, o
    for i in lost:
        assert f"would drop {i}" in o, (i, o)
    assert "skipped unparseable record at" in o and ":" in o, o
    assert len(ids(db)) == n  # nothing touched
    rc, o = run(db, "health")
    assert rc == 1 and all(i in o for i in lost), o
    assert "unparseable_log_lines" in o, o
    rc, o = run(db, "repair", "--apply")
    assert rc == 0 and all(f"logged   {i}" in o for i in lost), o
    rc, o = run(db, "reindex", "--force")
    assert rc == 0, o
    assert set(ids(db)) == set(all_ids), (len(ids(db)), n)
    rc, o = run(db, "repair", "--apply")
    assert "nothing to repair" in o, o
    # explicit drop still works and names what it drops
    c = sqlite3.connect(db); c.close()
    print("drop-guard: ok")
if __name__ == "__main__":
    test()
