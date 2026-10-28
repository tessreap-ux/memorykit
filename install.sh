#!/bin/sh
# memorykit installer. Checks Python and its pieces, installs what is missing
# (asks first unless --yes), puts the kit in its own venv, pins the model
# cache to a persistent folder, and runs `memorykit doctor`.
#
#   sh install.sh                 # root: /opt/memorykit, /usr/local/bin, /var/lib/memorykit/models
#                                 # user: ~/.local/share/memorykit, ~/.local/bin, ~/.cache/memorykit/models
#   sh install.sh --keyword-only  # no fastembed/numpy, no model (BM25 search only)
#   sh install.sh --no-model      # skip the model download (fetch later: memorykit doctor --fetch-model)
#   sh install.sh --yes           # install missing system packages without asking
#   sh install.sh --prefix DIR --bin-dir DIR --model-dir DIR
#
# Never touches any memory.db or memory-log. Uninstall: sh uninstall.sh
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
YES=0; KW=0; NOMODEL=0; PREFIX=""; BIN=""; MODELS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1;; --keyword-only) KW=1;; --no-model) NOMODEL=1;;
    --prefix) PREFIX=$2; shift;; --bin-dir) BIN=$2; shift;; --model-dir) MODELS=$2; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac; shift
done
if [ "$(id -u)" = 0 ]; then
  : "${PREFIX:=/opt/memorykit}" "${BIN:=/usr/local/bin}" "${MODELS:=/var/lib/memorykit/models}"
else
  : "${PREFIX:=$HOME/.local/share/memorykit}" "${BIN:=$HOME/.local/bin}" "${MODELS:=$HOME/.cache/memorykit/models}"
fi
say() { printf '%s\n' "memorykit: $*"; }
die() { printf '%s\n' "memorykit: ERROR: $*" >&2; exit 1; }

# -- system packages ---------------------------------------------------------
PKGS=""
need() { PKGS="$PKGS $1"; }
PY=""
for c in python3 python3.14 python3.13 python3.12 python3.11 python3.10 python3.9; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    PY=$(command -v "$c"); break
  fi
done
[ -n "$PY" ] || need python3
if [ -n "$PY" ]; then
  "$PY" -c 'import sqlite3' 2>/dev/null || need libsqlite3-0
  "$PY" -c 'import venv, ensurepip' 2>/dev/null || need python3-venv
fi
if [ -n "$PKGS" ]; then
  say "missing:$PKGS"
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
    if [ $YES = 0 ]; then
      printf 'memorykit: install with apt-get? [y/N] '; read -r a; case "$a" in y|Y|yes) ;; *) die "cannot continue without:$PKGS";; esac
    fi
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq $PKGS
  elif command -v dnf >/dev/null 2>&1; then
    die "install these with dnf (python3, python3-libs) and re-run"
  elif command -v apk >/dev/null 2>&1; then
    die "install these with: apk add python3 py3-pip sqlite-libs"
  else
    die "no supported package manager; install Python >= 3.9 with venv and sqlite3"
  fi
  for c in python3 python3.14 python3.13 python3.12 python3.11 python3.10 python3.9; do
    command -v "$c" >/dev/null 2>&1 && PY=$(command -v "$c") && break
  done
  "$PY" -c 'import sqlite3, venv, ensurepip' || die "Python is still missing sqlite3 or venv"
fi
say "python: $PY ($("$PY" -c 'import sys,sqlite3;print(sys.version.split()[0],"sqlite",sqlite3.sqlite_version)'))"

# -- venv + kit ----------------------------------------------------------------
# --system-site-packages: reuse fastembed/numpy if the system already has them
# (no download, no second 200 MB copy); anything missing goes into the venv.
mkdir -p "$PREFIX/lib" "$BIN"
[ -x "$PREFIX/venv/bin/python" ] || "$PY" -m venv --system-site-packages "$PREFIX/venv"
VPY="$PREFIX/venv/bin/python"
cp "$HERE/memorykit.py" "$PREFIX/lib/memorykit.py"
for f in schema.sql README.md DESIGN.md config.example.json; do [ -f "$HERE/$f" ] && cp "$HERE/$f" "$PREFIX/lib/"; done
[ -d "$HERE/templates" ] && cp -r "$HERE/templates" "$PREFIX/lib/"
if [ $KW = 0 ]; then
  if "$VPY" -c 'import fastembed, numpy' 2>/dev/null; then
    say "fastembed + numpy: already present"
  else
    say "installing fastembed + numpy into $PREFIX/venv (needs internet once)"
    "$VPY" -m pip install -q --disable-pip-version-check "fastembed>=0.3" numpy \
      || { say "WARN: pip install failed; continuing in keyword-only mode"; KW=1; }
  fi
fi

# -- model cache ---------------------------------------------------------------
mkdir -p "$MODELS"
OLD="${TMPDIR:-/tmp}/fastembed_cache"
if [ $KW = 0 ] && [ -d "$OLD" ] && [ -z "$(ls -A "$MODELS" 2>/dev/null)" ]; then
  say "reusing model files from $OLD (tmpfs) -> $MODELS"
  cp -a "$OLD"/. "$MODELS"/ 2>/dev/null || true
fi

# -- wrapper -------------------------------------------------------------------
W="$BIN/memorykit"
[ -L "$W" ] && rm -f "$W"
cat > "$W" <<EOF
#!/bin/sh
# memorykit wrapper (installed by install.sh)
export MEMORYKIT_MODEL_DIR="\${MEMORYKIT_MODEL_DIR:-$MODELS}"
$( [ $KW = 1 ] && echo 'export MEMORYKIT_NO_EMBED="${MEMORYKIT_NO_EMBED:-1}"' )
exec "$VPY" "$PREFIX/lib/memorykit.py" "\$@"
EOF
chmod 755 "$W"
printf '%s\n' "PREFIX=$PREFIX" "BIN=$W" "MODELS=$MODELS" > "$PREFIX/install.conf"

# -- verify ----------------------------------------------------------------------
if [ $KW = 0 ] && [ $NOMODEL = 0 ]; then
  "$W" --db "$PREFIX/doctor.db" doctor --fetch-model || die "doctor reported a failure"
else
  "$W" --db "$PREFIX/doctor.db" doctor || die "doctor reported a failure"
fi
case ":$PATH:" in *":$BIN:"*) ;; *) say "note: $BIN is not on PATH";; esac
say "installed $("$W" version) -> $W"
