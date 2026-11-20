import os,subprocess,sys,tempfile,sqlite3,glob,json
MK=os.path.join(os.path.dirname(os.path.abspath(__file__)),"memorykit.py")
def run(db,*a):
    r=subprocess.run([sys.executable,MK,"--db",db,*a],capture_output=True,text=True)
    return r.returncode,r.stdout+r.stderr
def count(db):
    c=sqlite3.connect(db);n=c.execute("select count(*) from memory_item").fetchone()[0];c.close();return n
def test():
    t=tempfile.mkdtemp();db=os.path.join(t,"x.db")
    run(db,"init")
    for i in range(5):
        rc,o=run(db,"capture","--speaker","user","--text",f"the deploy key number {i} lives in vault shelf {i*7} for project orchid",
                 "--no-auto") if False else run(db,"capture","--speaker","user","--text",f"the deploy key number {i} lives in vault shelf {i*7} for project orchid")
        assert rc==0,o
    n=count(db); assert n>=3,n
    ids=[r[0] for r in sqlite3.connect(db).execute("select id from memory_item order by learned_at")]
    # drift: drop 2 rows from DB (log-only), remove 1 log record (db-only)
    c=sqlite3.connect(db)
    for i in ids[:2]:
        c.execute("delete from memory_edge where src=? or dst=?",(i,i)) if False else None
        c.execute("delete from memory_item where id=?",(i,))
    c.commit();c.close()
    md=glob.glob(os.path.join(t,"**","*.md"),recursive=True)
    assert md,md
    lines=open(md[0]).read().split("\n")
    keep=[l for l in lines if ids[-1] not in l]
    open(md[0],"w").write("\n".join(keep))
    rc,o=run(db,"repair","--drift");assert "2 log-only" in o and "1 db-only" in o,o
    assert count(db)==n-2  # dry run changes nothing
    rc,o=run(db,"repair","--drift","--apply");assert rc==0,o
    assert count(db)==n,(count(db),o)
    c=sqlite3.connect(db);assert c.execute("select count(*) from memory_item where id=?",(ids[-1],)).fetchone()[0]==1
    assert any(ids[-1] in l for l in open(md[0]).read().split("\n"))
    assert glob.glob(db+".pre-repair-*.bak")
    rc,o=run(db,"repair","--drift","--apply");assert "nothing to repair" in o,o
    assert count(db)==n
    rc,o=run(db,"health");assert rc==0,o
    print("repair tests ok")
test()
