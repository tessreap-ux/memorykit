# memorykit on an iLands BYOA runner

The iLands runner (`ilands-runner`) launches a certified harness (Codex, Claude Code, ...) once per
heartbeat. The runner has no agent-callable start or end hook, so memorykit is wired in through the
agent's stable instructions. The model runs `pre-turn` first and `post-turn` last in every heartbeat.

## Layout

    ~/.ilands-runner/agents/<agent_id>/memory/
      memorykit.py   memory.db   memory-log/

Files the runner did not write survive runner restarts.

## Instructions to give the agent

Start of every heartbeat:

    python3 ./memory/memorykit.py --db ./memory/memory.db pre-turn --query "<task>" --budget 300 --prefer-source user

Before ending the heartbeat (keep it on ONE line):

    printf '%s\n' '{"speaker":"user","source":"user-said","session":"<run_id>","text":"<user message>"}' '{"speaker":"agent","session":"<run_id>","text":"<short summary>"}' | python3 ./memory/memorykit.py --db ./memory/memory.db post-turn --jsonl

## Pitfalls (found in a real install)

1. **Escaping inside a JS template literal.** The runner builds instructions in `harness/prompt.js`
   with template literals. Writing `\\\\` before a newline renders as `\\`. Bash reads that as a
   literal backslash plus end of command, so `printf` runs alone and nothing is stored. Render the
   string and run it against a scratch DB before you ship it.
2. **`AGENTS.md` is regenerated.** The runner rewrites `<workspace>/AGENTS.md` from its stable
   instructions on every heartbeat, so editing that file directly does not stick.
3. **Release upgrades drop the patch.** `prompt.js` lives under `releases/<version>/`. A runner update
   ships a fresh copy. Keep a patched copy next to the store and run `check-patch.sh` after every update.
4. **A skipped post-turn leaves no trace.** Journal replay only finishes a write that already started.
   If the model skips `post-turn` or the heartbeat is killed first, that turn is lost silently. A
   manifest and gap log to surface this are planned.

## Verify

    sh check-patch.sh                                   # OK / MISSING per installed release
    python3 ./memory/memorykit.py --db ./memory/memory.db health
