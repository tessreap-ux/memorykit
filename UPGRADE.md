# Updating memorykit in place (4.x -> 5.0)

Your data stays where it is. Nothing is exported, moved, or re-imported.

1. Replace the engine file. For an `install.sh` install that is
   `<prefix>/lib/memorykit.py` (check `head -5 $(which memorykit)` for the
   path); for a folder copy, overwrite `memorykit.py`. Copy `adapters/` beside
   it if you want the ready-made hooks.
2. Migrate the database where it lives:

       memorykit --db <your memory.db> upgrade

   This writes `<db>.pre-v5-<timestamp>.bak` first, adds the v5 state, switches
   SQLite to WAL, and checks integrity. Running it again is harmless.
3. Wire the two hooks into the agent loop (see README "The hook contract"):
   `pre-turn` before answering, `post-turn` after.
4. Remove any cron job or end-of-day call to `sleep`, `end-session`, `gc --apply`
   or `dedup --apply`. They still work by hand, but v5 does that work a slice
   per turn.
5. Check: `memorykit --db <your memory.db> health` should print `"problems": []`.

To roll back: put the 4.5 `memorykit.py` back and copy the `.bak` file over
the database. The markdown log in `memory-log/` is unchanged by the upgrade.
