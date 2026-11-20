# memorykit 4.5: integration brief (read this first)

This file is for whoever (human or AI) is wiring memorykit into an agent,
CLI, chatbot, or VM. README.md is the command reference; DESIGN.md explains
why each rule exists. Read DESIGN.md before changing any default.

## What it is
A local, single-file long-term memory layer for an agent.
- Source of truth: an append-only markdown log (`memory-log/YYYY-MM-DD.md`)
  plus any curated markdown you keep (templates/ has starters).
- Index: a SQLite file (`memory.db`) with vectors + BM25. Disposable;
  `reindex` rebuilds it from the markdown log.
- Engine: `memorykit.py`, Python stdlib only. Optional `fastembed` + `numpy`
  for semantic search (model BAAI/bge-small-en-v1.5, ~130 MB, downloads once).
  Without them it runs in keyword (BM25) mode.
- No server, no network calls except the one-time model download, no keys,
  no telemetry.

## Install
```sh
sh install.sh                 # full (vector) install, asks before apt-get
sh install.sh --keyword-only  # no ML deps, BM25 only (tiny, works anywhere)
memorykit doctor              # verify
```
Or with no installer at all: `python3 memorykit.py <command>` from this folder.
Or: `pip install .` / `pip install .[vector]` (console script `memorykit`).

Point it at a database with `--db PATH` or `MEMORYKIT_DB=PATH`.
One database per agent identity is the intended use.

## The integration contract (3 hooks)
Any host works if it can run a shell command (or import the module) at
these three points:

1. PRE-TURN (recall). Before the model answers, run:
   `memorykit search --query "<the user's message or a short topic>" --limit 5 --chars 200 --budget 300`
   Inject stdout into the model's context as "Relevant memory:". Output is
   one terse line per hit with a short id; empty output means nothing found.
   Make this automatic, not a tool the model may choose to skip.

2. POST-TURN (capture). After each exchange, capture both sides:
   `memorykit capture --speaker user  --provenance user-said  --session <S> --turn <N> --text "<user msg>"`
   `memorykit capture --speaker agent --provenance agent-said --session <S> --turn <N> --text "<reply>"`
   Tool results: `--speaker tool --provenance tool-observed`.
   The model's own guesses: `--provenance agent-inferred`.
   Pasted/quoted text from a third party: pass it via `--quote ... --quote-source NAME`
   so it is not stored as the speaker's own words.

3. SESSION END: `memorykit end-session --session <S>`
   (settles and distills that session's raw turns).

Corrections: capture the new fact with `--supersedes <old-id>` (or a shared
`--supersession-key state:<name>`). Never edit SQLite directly; changes
outside the commands are lost on rebuild.

Maintenance (event-triggered, dry-run by default; add `--apply` to act):
`gc`, `dedup`, `quarantine --scan`, `correlate --markdown MEMORY.md`, `health`.
`sleep` runs automatically on capture unless `MEMORYKIT_AUTO=0`.

## Host-specific notes
- Python host: `import memorykit` works, but the CLI is the stable
  interface; prefer `subprocess.run(["memorykit", ...], capture_output=True)`.
- Function-calling / tool-use agents (OpenAI, Gemini, Claude, local LLMs):
  expose `memory_search(query)` and `memory_capture(speaker, provenance, text)`
  as tools that shell out to the CLI, BUT still run hook 1 automatically.
- MCP: wrap the same two commands as MCP tools; no MCP server is bundled.
- Containers/VMs: mount the db + memory-log folder on a persistent volume;
  set the model cache dir (`install.sh --model-dir`) persistent too.
- Sandboxes without network: use `--keyword-only`.

## Binding rules (do not weaken)
- Never capture credentials, codes, keys, or private identifiers.
- Provenance firewall: agent-originated text cannot become a fact on its
  own; `promote` needs independent user-said or tool-observed evidence.
  The CLI refuses speaker=agent with provenance=user-said.
- No automated hard deletes. Quarantine is reversible; eviction is a human call.
- Contradictions are surfaced, never auto-resolved.

## Verify
`python3 test_memorykit.py` runs the suite in vector mode (if fastembed is
installed) then keyword mode. Packaged build: keyword 89/89 passed.

## Files
memorykit.py (engine) · schema.sql · config.example.json · install.sh ·
uninstall.sh · pyproject.toml · test_memorykit.py · templates/ ·
README.md · DESIGN.md · INTEGRATE.md (this file)
