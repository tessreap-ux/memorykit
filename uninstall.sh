#!/bin/sh
# Removes the memorykit program, venv and wrapper. Keeps every memory.db,
# memory-log, and the model cache (pass --models to remove the cache too).
set -eu
if [ "$(id -u)" = 0 ]; then C=/opt/memorykit/install.conf; else C=$HOME/.local/share/memorykit/install.conf; fi
[ -f "${1:-}" ] && C=$1
[ -f "$C" ] || { echo "memorykit: no install.conf at $C" >&2; exit 1; }
. "$C"
rm -f "$BIN"; rm -rf "$PREFIX"
[ "${1:-}" = "--models" ] || [ "${2:-}" = "--models" ] && rm -rf "$MODELS"
echo "memorykit: removed $PREFIX and $BIN"
