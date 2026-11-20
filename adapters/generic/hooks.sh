#!/bin/sh
# Generic two-hook wiring for any agent loop.  MK_DB must point at the store.
#   pre:  memorykit_pre  "<what the user just asked>"   -> prints memory to inject
#   post: memorykit_post "<user text>" "<agent reply>" -> stores both, one upkeep step
MK_DB="${MK_DB:-$HOME/.local/share/memorykit/memory.db}"
SESSION="${MK_SESSION:-$(date +%F)}"
memorykit_pre()  { memorykit --db "$MK_DB" pre-turn --query "$1" --budget "${MK_BUDGET:-250}" --prefer-source "${MK_USER:-user}"; }
memorykit_post() {
  python3 -c 'import json,sys; s,u,a,us=sys.argv[1:5]
print(json.dumps({"speaker":"user","source":us,"session":s,"text":u}))
print(json.dumps({"speaker":"agent","source":"agent","session":s,"text":a}))' \
    "$SESSION" "$1" "$2" "${MK_USER:-user}" | memorykit --db "$MK_DB" post-turn --jsonl
}
