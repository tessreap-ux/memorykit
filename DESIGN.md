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
plus manual override. A turn that differs from an existing one only
in numbers or negation ("costs 40" vs "costs 55") is a changed value,
not a restatement, so the novelty gate does not cap it.

### Retrieval: relevance first, salience only tilts

`relevance = (1-w) * minmax(cosine) + w * minmax(BM25)`, where `w` is
the share of the query's IDF mass that the best lexical match covers,
squared. Queries that use exact wording lean lexical; paraphrases lean
semantic. Salience multiplies relevance by 0.8 to 1.0, recency by 0.9
to 1.0. Neither can outrank relevance.

The old score (similarity x salience x recency) failed on a real
corpus: embedding similarities sit in a narrow band (0.55 to 0.65 for
bge-small), so a salience gap of 6 vs 8 outweighed relevance. On a
36-query benchmark from a 300-message multi-agent chat (24 queries that
share wording with the answer, 12 pure paraphrases):

| | lexical R@5 | paraphrase R@5 |
|---|---|---|
| old: sim x salience | 0.12 | 0.08 |
| pure cosine | 0.71 | 0.67 |
| new hybrid | 0.92 | 0.67 |
| keyword mode, BM25 + stopwords | 0.96 | 0.08 |

These are small, hand-written query sets, so treat them as direction,
not proof. Lexical matching drops common function words (STOPWORDS) from queries
and documents alike. Negations stay, because they change meaning. A
query made only of stopwords finds nothing in keyword mode instead of
everything. With stopwords removed, keyword-mode paraphrase recall is an
honest 0.08. The earlier 0.17 was partly lucky "the" matches. In vector
mode, nearly every row has some cosine, so the tail of a result list can
be unrelated. Ordering is right, and `--limit` and `--budget` cap the
cost. A raw-similarity floor would trim it if that's wanted.
A known weak spot: in keyword mode, BM25's length
normalization lets a long entry rank below short ones that share common
words. Run `bench` on your own corpus before tuning anything. Keyword mode can't do paraphrase; install the embedder if
that matters.

## Token economy

Everything the kit prints is read by someone who pays for it in
tokens. For agents whose tokens don't refresh, that is lifespan. So:

- **Search prints one line per hit:** a short id, the type, and a
  window of at most 200 characters (`--chars`) picked around the query
  words, or by cosine when the query is a paraphrase. Duplicate text is
  printed once. `--budget N` stops at about N tokens. `--verbose` adds
  scores. On the benchmark: about 557 tokens per search before, 263
  now, and about 52 with `--budget 80`. The answer was visible in the
  snippet for 27 of 30 found targets.
- **Ids print as 10 hex characters.** Any unique prefix of 6 or more
  characters is accepted wherever an id is taken.
- **`get` omits the embedding by default** (384 floats is thousands of
  tokens). `get --text` prints only the words.
- **Capture prints `saved id=… salience=…`.** Auto-sleep and auto-sweep
  are silent unless `MEMORYKIT_VERBOSE=1`. The missing-embedder note
  prints once per store. `health` is one compact line that omits zero
  fields (`--pretty` for all of them).
- **Distillation spends no tokens by default:** it keeps the text
  verbatim. An LLM distiller is opt-in.
- **Salience earned by use costs nothing:** a raw turn that search
  returns twice (`MEMORYKIT_SLEEP_MIN_HITS`) is distilled even if its
  phrasing scored under the floor. The hit counter lives in the DB, not
  the log, so a reindex resets it. That's a soft signal, by design.

## Conflict flags

When sleep writes a semantic entry that is near-identical (cosine 0.85
or higher, or string similarity in keyword mode) to a live one but
differs only in numbers or negation, it records
`possible_conflict_with` and `health` counts it. Nothing is resolved
automatically. Deciding which one is true takes evidence, not
similarity. Templated lines that differ only by a number ("Decision 0",
"Decision 1") will also be flagged. That's the price of catching real
value changes.

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

## Sleep: distillation triggered by function, not by clock

Raw turns are working memory. Once a turn has *settled*, it is distilled
into one semantic entry and the raw turn is wiped. Nothing runs on a
timer; the triggers are events:

- **Settled by distance:** a turn has left the working window once
  `MEMORYKIT_SETTLE_TURNS` (default 8) newer raw turns exist. A quick
  follow-up correction is no longer likely, so it's safe to distill.
- **Settled by closure:** `end-session --session S` settles every turn
  in that session at once.
- **Sweep by volume:** gc + dedup run after every
  `MEMORYKIT_SWEEP_EVERY` (default 50) distills.

With `MEMORYKIT_AUTO=1` (default) every capture checks for settled turns
and distills only those, so with the default verbatim distiller the
store stays distilled turn by turn. `sleep --apply` is the manual flush
(all eligible turns, optionally `--min-age-hours`); `--settled-only`
applies the same rule as auto.

Order, every time: **collect → distill → verify → wipe → sweep.**

- Turns below `MEMORYKIT_SLEEP_MIN_SALIENCE` (default 5) are not
  distilled; GC handles them.
- Distillation is 1:1: one live turn plus whatever it already superseded
  becomes one entry, linked by `distilled_from` edges. Those edges are
  not `derived_from`, so wiping a turn never cascades into its entry.
- The kit sets provenance, never the distiller: user-said or
  tool-observed → fact; agent → note. An optional `--distiller` command
  (or `MEMORYKIT_DISTILLER`) may rewrite the *text* only: JSON turns on
  stdin, `{"text": ...}` out. If it fails, the turn stays live.
- Wipe refuses unless the parent entry is live and every turn is linked
  to it. Wipe is a reversible tombstone, logged as `op: wipe` after it
  succeeds, so `reindex` keeps wiped turns wiped.
- A correction aimed at a wiped turn is redirected to its semantic
  entry, and supersession keys carry over, so distilled facts can still
  be corrected exactly.
- Crash safety: every step is idempotent. If a run dies after writing
  the entry but before the wipe, the next run finds the entry by its
  edge, re-logs it, and only wipes. If the wipe's log line was lost, a
  reindex revives the turn and the next run wipes it again. A lock file
  stops two runs from overlapping. It records PID, host, and time. A lock
  whose PID is gone on this host, or one older than
  `MEMORYKIT_LOCK_TIMEOUT`, is cleared by the next run, with a message.
  A held lock is reported even by auto-sleep, so a stall can't go
  unnoticed.
- `health` is read-only and reports what an unattended run can't see
  for itself: the settled backlog, entries written but not yet wiped,
  lock state, and rows that are in the DB but not the log (those would
  be lost on reindex).

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

- Keyword mode (no embedder): search is BM25; novelty gate
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
