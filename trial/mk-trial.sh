#!/bin/sh
# memorykit private trial, tests 1 and 3. Keyword mode. Scratch dir only.
# Never reads or writes your real docs. Output is a few short lines.
#   sh mk-trial.sh prep <dir-with-5-scratch-copies>   -> artifact path + sha256
#   (upload it with artifact-upload, delete local copy, download it cold)
#   sh mk-trial.sh verify <downloaded-file>            -> test 1 PASS/FAIL + test 3 RAM
set -eu
T=${MK_TRIAL_DIR:-$HOME/mk-trial}
PIN=v4.5.1
SHA=5ba216e41d9ee5abdb09eb824bff96fcdf76437bd0cfe79ba03fe5e9d0b053e6
export MEMORYKIT_NO_EMBED=1 MEMORYKIT_AUTO=0
mkdir -p "$T"
MK="$T/memorykit.py"
if [ ! -f "$MK" ]; then
  curl -fsSL "https://raw.githubusercontent.com/tessreap-ux/memorykit/$PIN/memorykit.py" -o "$MK"
fi
echo "$SHA  $MK" | sha256sum -c --quiet - || { echo "ABORT memorykit.py hash mismatch"; exit 2; }

case "${1:-}" in
prep)
  SRC=$2
  case "$(cd "$SRC" && pwd)" in "$T"*) ;; *) echo "note: copying from $SRC into scratch; originals untouched";; esac
  rm -rf "$T/a"; mkdir -p "$T/a/md"; cp "$SRC"/*.md "$T/a/" 2>/dev/null || true
  MEMORYKIT_MD_DIR="$T/a/md" python3 "$MK" --db "$T/a/a.db" init >/dev/null
  n=0
  for f in "$T"/a/*.md; do
    [ -f "$f" ] || continue
    # one capture per paragraph
    python3 - "$f" "$MK" "$T/a" <<'PY'
import sys, subprocess, os
f, mk, d = sys.argv[1:4]
env = dict(os.environ, MEMORYKIT_MD_DIR=d + "/md")
for i, p in enumerate(x.strip() for x in open(f, encoding="utf-8").read().split("\n\n")):
    if p:
        subprocess.run(["python3", mk, "--db", d + "/a.db", "capture", "--speaker", "agent",
                        "--text", p[:4000], "--session", os.path.basename(f), "--turn", str(i)],
                       env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
PY
  done
  N=$(python3 -c "import sqlite3,sys;print(sqlite3.connect(sys.argv[1]).execute('select count(*) from memory_item').fetchone()[0])" "$T/a/a.db")
  tar -czf "$T/mk-log.tgz" -C "$T/a" md
  echo "$N" > "$T/expected_rows"
  echo "rows=$N"
  echo "artifact=$T/mk-log.tgz"
  echo "sha256=$(sha256sum "$T/mk-log.tgz" | cut -d' ' -f1)"
  ;;
verify)
  F=$2
  [ -f "$F" ] || { echo "no file $F"; exit 1; }
  rm -rf "$T/b"; mkdir -p "$T/b"
  echo "sha256=$(sha256sum "$F" | cut -d' ' -f1)  (compare to prep)"
  tar -xzf "$F" -C "$T/b"
  [ -d "$T/b/md" ] || { echo "test1 FAIL no md/ in artifact"; exit 1; }
  python3 - "$MK" "$T/b" "$(cat "$T/expected_rows" 2>/dev/null || echo ?)" <<'PY'
import sys, subprocess, resource, sqlite3, time
mk, d, exp = sys.argv[1:4]
t = time.time()
r = subprocess.run(["python3", mk, "--db", d + "/b.db", "reindex", "--md-dir", d + "/md", "--force"],
                   capture_output=True, text=True)
dt = time.time() - t
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
n = sqlite3.connect(d + "/b.db").execute("select count(*) from memory_item").fetchone()[0] if r.returncode == 0 else -1
ok = r.returncode == 0 and str(n) == exp
print(f"test1 {'PASS' if ok else 'FAIL'} rows={n} expected={exp}")
if r.returncode: print("err:", r.stderr.strip()[-300:])
print(f"test3 reindex peak_rss={peak}MB time={dt:.1f}s mode=keyword")
PY
  ;;
*) sed -n 2,7p "$0"; exit 1;;
esac
