# Vector-memory map kit

A plug-and-play memory system for any agent. No content included —
you bring your own memories; this is the machinery.

## What it is

Three layers, one rule: **markdown is the source of truth, the vector
index is a disposable map of it.**

- `MEMORY.md` + daily logs + people/group pages: curated, hand-edited,
  authoritative. Templates in `templates/`.
- `memory.db`: a SQLite vector index that answers "what did we already
  know about X?" by meaning, not keyword. Rebuildable any time.
- `memorykit.py`: the whole engine in one file — capture, search,
  supersede, dedup, garbage collection, cross-layer audit.

Read `DESIGN.md` before changing the defaults. It explains the five
cascade failure modes this design defends against, and why each rule
exists.

## Requirements

- Python 3.10+ (standard library only for the core).
- For vector search: `pip install fastembed` — the embedding model
  (default `BAAI/bge-small-en-v1.5`) downloads once on first use.
  Without it, capture still works, search falls back to keyword
  matching, and `backfill` computes vectors later.

No server, no account, no key, no sync. Local-only by default.

## Quickstart

```bash
# 1. Create the empty index
python3 memorykit.py init

# 2. Capture turns (provenance is mandatory — see DESIGN.md)
python3 memorykit.py capture --speaker user --text "The photo folder is called Private"
python3 memorykit.py capture --speaker agent --text "I think the folder name matters" \
    --provenance agent-inferred --salience 6

# 3. Search before answering (automatic, never discretionary)
python3 memorykit.py search --query "photo folder name" --limit 5

# 4. Correct by superseding, never by silent overwrite
python3 memorykit.py capture --speaker user --text "Correction: the folder is called Quiet" \
    --supersedes <old-id>

# 5. Nightly tidy (dry-run first, always)
python3 memorykit.py gc            # dry run
python3 memorykit.py gc --apply    # quarantine below-floor entries
python3 memorykit.py dedup         # dry run
python3 memorykit.py dedup --apply
python3 memorykit.py correlate --markdown MEMORY.md
```

## Commands

| Command | What it does |
|---|---|
| `init` | Create the empty database from `schema.sql`. |
| `capture` | Store a turn with provenance, salience, confidence, novelty gate, supersession wiring. |
| `search` | Meaning-based recall over live entries only (superseded and quarantined rows are filtered out). |
| `get` | Show one entry in full, including metadata. |
| `supersede` | Point an old entry at its replacement; old row kept, filtered from search. |
| `dedup` | Merge near-duplicates (three gates + identifier guard). Dry-run by default. |
| `gc` | Score every live entry; quarantine below-floor ones. `--restore ID` undoes one; `--eviction-report` lists quarantine-expired candidates for manual review. |
| `backfill` | Compute vectors for rows stored before an embedder was installed. |
| `correlate` | Audit curated markdown claims against the index; orphans reported, never deleted. |

Options: `--db PATH` (default `memory.db`, or the `MEMORYKIT_DB`
environment variable), `--session`, `--turn`, `--supersession-key
state:foo` (auto-retires prior versions with the same key),
`--parent-ids` (writes `derived_from` edges for cascade pruning),
`--check` (advisory contradiction search before capture).

## Rebuilding

The database is an index, not an archive. If it disagrees with the
markdown, the markdown wins:

```bash
rm memory.db
python3 memorykit.py init
# re-capture from your markdown corpus
```

## Safety rules (binding)

- Never capture credentials, codes, key material, or private
  identifiers. This is not a vault.
- Never hard-delete by automation. Quarantine is reversible; eviction
  is a human decision.
- Never let agent-originated content promote itself to fact.
  Promotion needs independent evidence: the user restating it, or a
  tool observing it.
- Never auto-resolve a contradiction. Surface it, supersede
  explicitly.

## Files

```
memorykit.py            the engine (one file)
schema.sql              the index schema
DESIGN.md               the design: layers, failure modes, defenses
README.md               this file
config.example.json     tunable thresholds (copy to config.json if you wire one in)
templates/
  MEMORY.md             curated memory template
  daily-log.md          daily log template
  WORKING-MEMORY.md     shared group working-memory template
```
