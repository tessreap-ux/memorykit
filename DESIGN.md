# Vector-memory map design

The design in one sentence: **markdown is the territory, the vector
index is a disposable map of it, and every entry carries the metadata
needed to keep the map from corrupting the territory.**

## Layers

1. **Curated markdown (source of truth).** A `MEMORY.md` plus dated daily
   logs plus per-person / per-group pages. Human- and agent-readable,
   hand-edited, never auto-rewritten. If every other layer burns, this
   one rebuilds them.
2. **Daily logs.** Raw day-to-day detail lives here; only durable facts
   get promoted into `MEMORY.md`. This keeps the curated layer tight.
3. **Vector index (the map).** A SQLite file holding embedded turns and
   facts. It exists to answer "what did we already know about X?" by
   meaning, not keyword. It is rebuildable from layer 1 at any time:
   delete the database, re-index the markdown. Nothing in it is
   authoritative on its own.
4. **Working memory (optional, group use).** A shared scratch document
   for a group's live thinking — findings, hypotheses, open questions,
   decisions — so the thinking doesn't have to live in anyone's
   personal memory. See `templates/WORKING-MEMORY.md`.

## The problem: cascade memory failure

A memory system that writes every turn corrupts itself unless each
entry carries metadata defending against five failure modes:

1. **Write-back corruption** — a slightly-wrong summary is re-ingested
   as ground truth; the error compounds every retrieval cycle.
2. **Echo reinforcement** — a retrieved memory shapes the response,
   the response is indexed, the same content is reinforced; the record
   drifts from truth.
3. **Repetition-as-confirmation** — confidence inflates through
   self-repetition instead of independent evidence.
4. **Contradiction pile-up** — stale entries contradict fresh ones;
   retrieval returns both as "fact".
5. **Trivia drowning** — uniform indexing buries signal in noise.

## The defenses (binding on capture and retrieval)

### Provenance is mandatory
Every entry is stamped `user-said`, `tool-observed`, `agent-said`, or
`agent-inferred`. Provenance travels with the entry at retrieval time:
an `agent-inferred` memory is cited as inference, never as fact.

### Provenance firewall
`agent-said` / `agent-inferred` entries are stored as `note`, never
`fact`. They can only be promoted by *independent* evidence: the user
restating it, or a tool observing it. The agent's own repetition is
not evidence.

### Store the text, not a clever summary
Light structuring is fine. Aggressive summarization is where
corruption enters (failure mode 1).

### Confidence starts honest
`user-said` / `tool-observed` = 1.0, `agent-said` = 0.7,
`agent-inferred` = 0.4. Confidence rises ONLY on independent evidence
— never because the entry was retrieved and repeated (modes 2, 3).

### Corrections supersede, never silently overwrite
A correction points at the entry IDs it replaces (`superseded_by`).
The old row stays as an audit trail but is filtered from retrieval by
default (mode 4). Supersession keys (`state:foo`) auto-retire prior
active entries carrying the same key, so only the canonical version
stays retrievable.

### Novelty gate (capture)
A new turn is compared against the top live entries. A verbatim or
near-verbatim restatement (cosine > 0.90 or text similarity > 0.85)
has its salience capped at 3, so repetition can never inflate into
importance. A genuinely novel turn (best cosine < 0.50) gets +1.
The gate never fails a capture; on any error the heuristic salience
stands.

### Promotion needs independent evidence
`promote` turns a note into a fact only when pointed at a different,
live, `user-said` or `tool-observed` entry that is not derived from the
note (checked transitively). The link is stored as a `supported_by`
edge. Speaker `agent` cannot self-stamp user or tool provenance.

### Contradiction check, never auto-resolve
For important turns, run a similarity search first and review the
candidates. The agent decides and supersedes explicitly. The system
never auto-resolves a contradiction.

### Skip trivia, score the rest
Turns under 10 characters are skipped. Salience 1–10 is a transparent
heuristic (decisions, corrections, questions, length, emotion words)
plus manual override. Retrieval weights similarity x salience x
recency.

## Entity resolution (dedup)

Near-duplicates merge only when ALL gates pass:

- same entry type,
- embedding cosine >= 0.97,
- raw-text similarity >= 0.90.

Candidates are blocked by a vectorized cosine pass first; difflib only
runs on pairs that clear the cosine gate.

Plus a semantic guard: if two entries differ ONLY in identifier-like
tokens (timestamps, message ids, counts), they are distinct historical
events and are NEVER merged — embeddings alone cannot tell "same event
twice" from "same template, different events". Single-linkage
clustering is banned for the same reason: it chains recurring
templated entries into one blob.

Merges keep the highest-provenance entry and supersede the rest.
Dedup is reversible tombstoning, never deletion.

## Garbage collection (the nightly tidy)

The store is self-healing, not append-only. Every GC pass scores each
live entry:

    S = U x e^(-lambda x dt) x R

- **U (utility):** confidence x (salience / 10), adjusted by explicit
  feedback (+1 ratings x1.5 each capped at x3, -1 ratings x0.5 floored
  at x0.25). Honest limit: this approximates "did retrieval help" —
  no causal instrument exists here, and none is pretended.
- **Decay:** lambda = ln(2) / half-life, measured from `learned_at`.
  Retrieval never resets decay; access only feeds R (capped at 1.5x). Half-lives: low-salience
  turns 7d, notes 30d, facts 180d, rules 365d.
- **R (reinforcement):** 1 + 0.5 x e^(-days_since_access / 30).

S below the noise floor (0.05) → **quarantined** (status `rejected`,
reversible soft-delete, never hard-deleted by the GC). Quarantined
entries sit 30 days; `--eviction-report` lists expired candidates for
manual review. `--restore` pulls one back. Cascade pruning follows
`derived_from` edges recursively, but never quarantines an exempt
child; it flags `orphaned_parent` instead. `--restore` undoes the
cascade it caused and restarts the decay clock (`restored_at`), so the
next GC doesn't undo the restore; `learned_at` keeps the true age. High-salience rules (salience >= 9)
and `metadata.protected` entries are exempt. Default is dry-run;
`--apply` executes.

Auto-restore-on-regression is deliberately NOT implemented: no
reliable "reasoning regression" signal exists. Manual `--restore`
instead of a fake automatic one.

## Cross-layer audit (correlate)

Every curated-markdown claim is checked against the vector index:
ok / orphan / stale. Orphans and stale claims are REPORTED, never
auto-deleted — the curated layer outranks the index, so an orphan may
mean the index is stale, not that the claim is wrong.

## Rebuilding

Capture and every state change also append to a dated markdown log
with one machine-readable record per line. `reindex` replays that log
in timestamp order through the same code paths, so statuses,
supersessions, feedback, promotions, learned_at, restore times and
quarantine times (so decay and eviction aging) come back exactly:

```bash
rm memory.db
python3 memorykit.py reindex
```

## Mode differences (stated, not hidden)

- Keyword mode (no embedder): search is token overlap; novelty gate
  relies on text similarity only; dedup merges only exact normalized
  text; correlate uses a 0.3 overlap cutoff.
- Vector mode: cosine scales differ (bge-small scores unrelated English
  around 0.45), so correlate uses 0.75. This cutoff was calibrated on a
  small sample; tune it on your own corpus.

The database is an index, not an archive. If it disagrees with the
markdown, the markdown wins.

## What this design deliberately does NOT do

- No sync, no account, no key, no server. Local-only by default.
- No auto-promotion of agent content to fact.
- No auto-resolution of contradictions.
- No hard deletes by automation.
- No capture of credentials, codes, key material, or private
  identifiers — the capture path is not a vault.
