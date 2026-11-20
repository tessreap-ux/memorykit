# memorykit 5.0: integration brief (read this first)

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

## The integration contract (2 hooks, no session end)
Any host works if it can run a shell command (or import the module) at
these two points. There is no third hook and no nightly job.

1. PRE-TURN (recall). Before the model answers:
   `memorykit pre-turn --query "<the user's message>" --budget 250 --prefer-source <user name>`
   Inject stdout as "Relevant memory:". One terse line per hit with a short
   id. Make it automatic, not a tool the model may skip.

2. POST-TURN (capture + upkeep). After the model answers, send both sides
   as JSON lines in one call:
   `printf '%s\n' '{"speaker":"user","source":"<name>","session":"S","turn":"N","text":"..."}' '{"speaker":"agent","session":"S","turn":"N","text":"..."}' | memorykit post-turn --jsonl`
   Tool results: `"speaker":"tool"`. The model's own guesses:
   `"provenance":"agent-inferred"`. Third-party pasted text: `"quote":"...","quote_source":"NAME"`.
   Python: `memorykit.pre_turn(db, q, budget)` / `memorykit.post_turn(db, turns)`.

`end-session` still exists as an optional hint; nothing depends on it.

Corrections: capture the new fact with `--supersedes <old-id>` (or a shared
`--supersession-key state:<name>`). Never edit SQLite directly; changes
outside the commands are lost on rebuild.

Maintenance runs a bounded slice inside every post-turn (MEMORYKIT_AUTO=1,
the default). Manual audits, dry-run by default: `gc`, `dedup`,
`quarantine --scan`, `correlate --markdown MEMORY.md`, `health`.

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
installed) then keyword mode; `python3 test_v5.py` runs the per-turn,
crash and upgrade checks.

## Files
memorykit.py (engine) · schema.sql · config.example.json · install.sh ·
uninstall.sh · pyproject.toml · test_memorykit.py · templates/ ·
test_v5.py · adapters/ · README.md · DESIGN.md · INTEGRATE.md (this file) ·
UPGRADE.md · CHANGELOG.md · LICENSE
