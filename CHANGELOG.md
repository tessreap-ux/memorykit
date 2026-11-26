# Changelog

## 5.0.2

- `reindex --force` refuses (exit 2) when DB rows are missing from the log and names each one (`would drop <id>`). `--drop-missing` rebuilds anyway and names each dropped row. Reported by Miso.
- Torn or unparseable log lines are reported as `skipped unparseable record at <file>:<line>` by reindex, repair and upkeep; `health` lists them.
- `health` lists ids (`in_db_not_in_log_ids`, `in_log_not_in_db_ids`, first 100), not just counts.
- `repair` prints every id it finds, restores and logs.
- `upkeep` runs drift repair when the DB and log disagree (`--no-repair` to skip).
- New test: test_drop_guard.py (Miso's damaged-log rig).
- adapters/ilands-runner: wiring notes for iLands BYOA runners, `check-patch.sh`, and pitfalls found in a real install (template-literal escaping, regenerated AGENTS.md, release upgrades, silent skipped post-turn). Credit: Agy for the first install.

## 5.0.1 (2026-11-20)

### Added
- `memorykit repair --drift [--apply]`: heals drift between the DB and the markdown
  log without dropping anything. Dry run by default. With `--apply` it backs the DB up
  (`<db>.pre-repair-<ts>.bak`), writes DB-only rows (and their provenance) into the log,
  replays log-only captures into the DB, and re-applies the supersede, quarantine and
  relate records that touch them. Idempotent; a second run reports nothing to repair.
  Edges and weights on existing nodes are never rewritten. (Reported from a live store that had
  300 log-only entries and 1 DB-only row; a plain `reindex` would have lost it.)
- `test_repair.py`.

## 5.0.0 (2026-11-19)

The end-of-day maintenance is gone. Everything happens per turn.

### Added
- `pre-turn` / `post-turn` hooks (CLI) and `pre_turn()` / `post_turn()` (Python):
  the whole integration is two calls.
- Write-ahead capture journal (`<db>.journal.jsonl`, fsynced). Any call replays
  unfinished turns, so a crash at any point loses nothing and duplicates nothing.
- Deterministic turn ids: retrying a turn is a no-op.
- Bounded upkeep slice on every `post-turn`: distill at most 2 settled turns,
  GC-score 25 rows and dedup-check 8 rows from rolling cursors. Tunable with
  `MEMORYKIT_UPKEEP_DISTILL`, `MEMORYKIT_UPKEEP_GC_ROWS`, `MEMORYKIT_UPKEEP_DEDUP_ROWS`.
- Turns settle by age (`MEMORYKIT_SETTLE_HOURS`, default 6) as well as by count,
  so a session nobody closes still drains.
- `upgrade`: in-place migration from 4.x with an automatic backup; idempotent.
- `upkeep`: run slices by hand (never required).
- `--prefer-source` lane in recall (two reserved slots for the user's own words).
- `adapters/`: generic shell hooks, a chat-journal batch adapter, notes for a
  second agent.
- SQLite WAL mode with `synchronous=FULL`.
- `health` reports `journal_pending` and `upkeep_last`.
- `test_v5.py`: real SIGKILL-style crash tests and missed-session tests.

### Changed
- `capture` runs one bounded upkeep slice instead of the old auto-sleep plus a
  full gc + dedup sweep every 50 distills.
- `capture` with a forced id is idempotent (also makes `reindex` replays safe).
- `end-session` and `sleep` are optional manual tools; nothing depends on them.

### Removed
- `MEMORYKIT_SWEEP_EVERY` and the `distills_since_sweep` counter.

## 4.5.0
Embedded schema, installer, `doctor`, graph operations (`relate`, `weigh`,
`merge`, `quarantine` by degree).
