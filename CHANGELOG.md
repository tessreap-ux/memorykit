# Changelog

## v4.5.1
- `INTEGRATE.md` adds a 3-hook contract for any agent: pre-turn search, post-turn capture, end-session.
- Test fix for dynamic aging.

## v4.5
- Schema is embedded, so installs no longer need `schema.sql` next to the script.
- `install.sh` and `uninstall.sh` check dependencies; `doctor` checks health.
- Persistent model cache via `MEMORYKIT_MODEL_DIR`.
- Graph ops: weighted `relate`, node `weigh`, `merge`, and quarantine by degree.
- Additive DB migration for edge weights.

## v4.3 – v4.4
- Attribution with `--source` and `--quote`.
- Scale caches: search over 20k memories went from 9.7 s to 0.7 s.
- Held-out benchmark of 30 pairs: R@5 rose from 0.30 to 0.77.

## v4 – v4.2
- Hybrid cosine + BM25 ranking, with salience as a tilt rather than a multiplier.
- Token-lean output, short ids, and `get --text`.
- Use-earned distillation, a novelty fix for changed values, conflict flags, and a `bench` command.

## v3 – v3.3
- Sleep triggered by function rather than clock: settle after 8 newer turns or at end-session, and sweep every 50 distills.
- Distiller output is stripped of forged provenance headers.
- Stale-lock recovery, NFKC header stripping, and a `health` command.

## v2 – v2.1
- Reindex from the markdown log, plus `feedback`, `promote`, a salience regex and scalable dedup.
- `restore` restarts decay; `rebuild` keeps quarantine times.

## v1.1
- Patched the original kit: 6 bug fixes.
