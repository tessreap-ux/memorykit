# memorykit

**Long-term memory for any agent, in one file, with no nightly job.**

memorykit gives an agent a memory that survives its context window: it
stores every turn with provenance, recalls what matters at the start of the
next turn within a token budget, and keeps itself tidy a little at a time.
One SQLite file, an append-only markdown log, Python standard library only.
Vectors are optional; without them it runs on BM25 keyword search.

```
            ┌─────────────────────────── one turn ───────────────────────────┐
            │                                                                │
 user msg ──►  pre_turn(query)                      post_turn(turns)         │
            │   1. replay journal  (finish any       1. append to journal    │
            │      write a crash left half-done)        + fsync              │
            │   2. hybrid search  (BM25 + cosine)    2. store, idempotent    │
            │   3. trim to token budget                 (novelty, dedup id,  │
            │          │                                 supersede, edges)   │
            │          ▼                              3. ONE bounded upkeep  │
            │   inject ≤ N tokens ──► agent answers ──►  step:               │
            │                                            • distill ≤2 turns  │
            │                                            • GC-score 25 rows  │
            │                                            • dedup-check 8 rows│
            └────────────────────────────────────────────────────────────────┘
                 memory.db (SQLite, WAL)   memory-log/*.md (source of truth)
```

## Why v5

Earlier versions leaned on end-of-session passes: `sleep`, `end-session`,
`gc --apply`, `dedup --apply`. If the agent crashed, went offline, or simply
never sent "end of day", the work never ran. v5 removes that dependency.

* **Every bit of upkeep happens inside the turn.** Each `post_turn` runs one
  small, bounded slice of maintenance from a rolling cursor. The cost is the
  same whether the last call was a minute or a month ago.
* **Decay is never "run".** Scores are computed from timestamps when a row is
  looked at, so there is nothing to catch up on.
* **Crash-safe.** Turns are written to a fsynced journal before the database.
  Kill the process at any point and the next call, whatever it is, finishes
  the write exactly once. Turn ids are deterministic, so a retried turn never
  duplicates.
* **One writer at a time, no stale locks.** Writes take a kernel `flock` on
  `<db>.lock`, which the OS releases if the process dies.
* **Nothing waits for a session to end.** A raw turn settles after 8 newer
  turns or 6 hours, whichever comes first. `end-session` still exists; it is
  only a hint.

## Quickstart

```sh
# clone this repo, or just download memorykit.py: it is the whole engine
python3 memorykit.py --db memory.db init          # stdlib only: BM25 mode
pip install fastembed numpy                        # optional: semantic search
```

Wire two hooks into your agent loop:

```sh
# start of turn: what should I remember about this?
python3 memorykit.py --db memory.db pre-turn --query "$USER_MSG" --budget 250 --prefer-source user

# end of turn: store what was said (batch, one process, one model load)
printf '%s\n' \
  '{"speaker":"user","source":"user","text":"I moved the standup to 9:30"}' \
  '{"speaker":"agent","text":"Noted, standup is 9:30 now."}' \
| python3 memorykit.py --db memory.db post-turn --jsonl
```

Or in Python:

```python
import memorykit as mk
context = mk.pre_turn("memory.db", user_msg, budget=250, prefer_source="user")
# ... the agent answers, with `context` in its prompt ...
mk.post_turn("memory.db", [
    {"speaker": "user", "source": "user", "text": user_msg},
    {"speaker": "agent", "text": reply},
])
```

Ready adapters live in `adapters/`: a generic shell pair (`generic/hooks.sh`)
and a batch adapter that turns a chat-log export (JSON lines of
`{id, role, time, text}`) into one `post_turn` call (`chatlog/`).

## The hook contract

| Hook | When | Input | Output | Guarantees |
|---|---|---|---|---|
| `pre_turn` | before the agent answers | query, token budget, optional preferred speaker | terse lines, one per memory, ≤ budget | read-mostly; repairs any interrupted write first |
| `post_turn` | after the agent answers | list of turns: `speaker` (user/agent/tool), `text`, optional `source`, `session`, `turn`, `provenance`, `salience`, `supersedes`, `supersession_key`, `protected` | stored ids | journaled + fsynced first; idempotent; one bounded upkeep step |

Provenance is enforced, not advisory: an agent cannot store its own words as
`user-said` or `tool-observed`.

## Upgrading an existing install

```sh
memorykit --db path/to/memory.db upgrade
```

Backs up the database next to itself, adds what v5 needs, switches SQLite to
WAL, and finishes any pending journal. Rows are not moved or rewritten, and
running it twice is harmless. Then delete any cron or end-of-day call to
`sleep` / `end-session`; nothing depends on them any more.

## Commands

| Command | What it does |
|---|---|
| `pre-turn` | Start-of-turn recall within `--budget` tokens; `--prefer-source NAME` reserves two slots for that speaker. |
| `post-turn` | End-of-turn store. `--jsonl` reads a batch on stdin; or `--speaker/--text`. `--no-upkeep` skips the upkeep slice. |
| `upgrade` | In-place migration from any 4.x database (backs up first). |
| `upkeep` | Run N upkeep slices now. Never required. |
| `init`, `capture`, `search`, `get` | The lower-level primitives the hooks are built on. |
| `supersede`, `feedback`, `promote`, `relate`, `weigh`, `merge`, `quarantine` | Corrections and graph operations; all reversible. |
| `gc`, `dedup`, `sleep`, `end-session` | Full-table passes from 4.x. Kept for manual audits; never required. |
| `health`, `doctor`, `bench` | Read-only checks: backlog, journal, log/DB drift, deps, recall@k. |
| `reindex`, `backfill`, `correlate` | Rebuild from markdown, add vectors later, audit curated notes. |

## Benchmarks

Measured on an 8-thread Linux netbook (7.6 GB RAM), a real 845-row store,
bge-small-en-v1.5 via fastembed:

| Operation | v4.5 | v5 |
|---|---|---|
| store 10 turns (CLI) | 13.3 s (10 processes) | 3.1 s (one `post-turn --jsonl`) |
| store 1 turn, in-process | — | 0.12 s |
| recall, in-process (250-token budget) | — | 0.20 s |
| one upkeep slice | full-table `sleep` + `gc` + `dedup` | 0.55 s, bounded |
| `upgrade` of the 845-row store | — | 0.22 s |

Test suites, all passing in keyword and vector mode: 22 v5 checks
(kill-mid-turn, kill-before-log, torn journal, retry, missed session,
amortized GC and dedup, token budget, provenance firewall, in-place upgrade,
concurrent replay by three processes, Python API) plus the 89 carried-over
4.x checks.

## Design

* Markdown is the source of truth; the database is a disposable index
  (`reindex` rebuilds it).
* Soft-delete only. Automation never hard-deletes; quarantine is reversible.
* Salience and recency tilt ranking; they never outrank relevance.
* No server, no network except the one-time model download, no telemetry.

See `DESIGN.md` for the failure modes this guards against and why each rule
exists, and `CHANGELOG.md` for what changed.

## License

MIT
