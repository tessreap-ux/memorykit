#!/usr/bin/env python3
"""memorykit — a plug-and-play vector-memory map for any agent.

One self-contained engine implementing the vector-memory map design:

  * Markdown is the source of truth. The vector index is a disposable map
    that can be rebuilt from markdown at any time.
  * Every entry carries provenance, salience, confidence, and supersession
    links, so the store cannot corrupt itself by re-ingesting its own
    output (see DESIGN.md for the five cascade failure modes).
  * Soft-delete only. Nothing is ever hard-deleted by automation;
    quarantine is reversible, eviction is a human decision.

Storage: a single SQLite file (default ./memory.db). Embeddings live in
the `embedding` column as a JSON list of floats. No server, no account,
no sync, no key.

Vector search needs an embedder. If `fastembed` is installed, vectors are
computed locally with the configured model (default bge-small-en-v1.5,
downloaded once on first use). If no embedder is available, capture still
stores everything, search falls back to keyword matching, and `backfill`
computes vectors later once an embedder is installed.

Usage:
  python3 memorykit.py init [--db PATH]
  python3 memorykit.py capture --speaker user|agent|tool --text "..."
      [--salience N] [--supersedes id,id] [--supersession-key state:foo]
      [--parent-ids id,id] [--check] [--project NAME]
  python3 memorykit.py search --query "..." [--limit N]
  python3 memorykit.py get --id ID
  python3 memorykit.py supersede --old ID --new ID
  python3 memorykit.py dedup [--apply]        # report, or merge + supersede
  python3 memorykit.py gc [--apply]           # dry-run by default
  python3 memorykit.py gc --restore ID
  python3 memorykit.py gc --eviction-report
  python3 memorykit.py backfill
  python3 memorykit.py correlate --markdown MEMORY.md
  python3 memorykit.py feedback --id ID --rating +1|-1 [--comment "..."]
  python3 memorykit.py promote --id NOTE_ID --evidence EVIDENCE_ID
  python3 memorykit.py reindex [--md-dir DIR] [--force]
  python3 memorykit.py pre-turn --query "..." [--budget N] [--prefer-source NAME]
  python3 memorykit.py post-turn --jsonl < turns.jsonl      # or --speaker/--text
  python3 memorykit.py upgrade                # in-place migration from 4.x
  python3 memorykit.py sleep [--apply] [--min-age-hours H] [--distiller CMD]
  python3 memorykit.py end-session --session S
  python3 memorykit.py health

Every capture and state change is also appended to ./memory-log/YYYY-MM-DD.md
(override with MEMORYKIT_MD_DIR), which is what `reindex` rebuilds from.
"""
import argparse
import contextlib
import datetime
import io
import time
import unicodedata
import difflib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import uuid

DEFAULT_DB = os.environ.get("MEMORYKIT_DB", "memory.db")
MODEL = os.environ.get("MEMORYKIT_MODEL", "BAAI/bge-small-en-v1.5")
# Where the embedding model is cached. fastembed's own default is
# $TMPDIR/fastembed_cache, which tmpfs wipes on reboot (then search silently
# drops to keyword mode until the model is downloaded again).
MODEL_DIR = os.environ.get("MEMORYKIT_MODEL_DIR") or os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
    "memorykit", "models")
VERSION = "5.0.5"

# Provenance -> (entry type, default confidence). Agent-originated content
# is NEVER stored as fact: only user statements and tool observations earn
# that. Agent content can only be promoted by independent evidence (the
# user restating it, or a tool observing it). Repetition is not evidence.
PROVENANCE = {
    "user-said": ("fact", 1.0),
    "tool-observed": ("fact", 1.0),
    "agent-said": ("note", 0.7),
    "agent-inferred": ("note", 0.4),
}
SPEAKER_PROVENANCE = {
    "user": "user-said",
    "tool": "tool-observed",
    "agent": "agent-said",
}

# Decision markers are phrases, not single common verbs: a bare "will"
# matched nearly every sentence and handed out +2 to everything.
DECISION_RE = re.compile(
    r"\b(decided|decision|we agreed|agreed to|i prefer|prefers?\b|preference"
    r"|always|never|must\b|must not|rule\b|rule:|boundary|promised?|promise to"
    r"|commit(?:ted|ment)?\b|from now on|going forward|standing (?:rule|order)"
    r"|won't ever|will never|do not ever|deadline|by (?:monday|tuesday|wednesday"
    r"|thursday|friday|saturday|sunday))",
    re.I)
CORRECTION_RE = re.compile(
    r"\b(wrong|actually\b|correction|mistaken|misquot|not a|isn't|aren't)\b", re.I)
EMOTION_RE = re.compile(r"\b(love|grief|fear|angr|hate|devot|hold me)\b", re.I)

# Dedup gates (all must pass to merge): same type, embedding cosine >=
# 0.97, raw-text similarity >= 0.90. The identifier guard blocks merges
# whose ONLY differences are identifier-like (timestamps, message ids,
# counts) — those are distinct historical events, not duplicates.
DEDUP_COSINE = 0.97
DEDUP_TEXT = 0.90
# Novelty gate thresholds (capture): restatement caps salience at 3,
# genuinely novel turns get +1.
NOVELTY_RESTATEMENT_COS = 0.90
NOVELTY_RESTATEMENT_TXT = 0.85
NOVELTY_NOVEL_COS = 0.50
# GC: score below this floor quarantines the entry (reversible).
GC_NOISE_FLOOR = 0.05
# Correlate: an orphan is a claim whose best neighbor is below threshold.
# Keyword overlap and embedding cosine live on different scales: bge-small
# scores unrelated English pairs around 0.45, so 0.3 never flags anything.
CORRELATE_KEYWORD_MIN = 0.3
CORRELATE_COSINE_MIN = float(os.environ.get("MEMORYKIT_CORRELATE_COS", "0.75"))
GC_QUARANTINE_DAYS = 30
# Distill: raw turns at or above this salience are distilled
# into semantic entries and then wiped; below it they are left for GC.
SLEEP_MIN_SALIENCE = int(os.environ.get("MEMORYKIT_SLEEP_MIN_SALIENCE", "5"))
SEMANTIC_PROJECT = "semantic"
# Function-based triggers (no clock). With MEMORYKIT_AUTO=1 (default) every
# post-turn (and every legacy capture) runs one bounded upkeep step (see the
# v5 section). A raw turn is SETTLED, and may be distilled, when:
#   * SETTLE_TURNS newer captures have arrived since it (it has left the
#     working window, so a quick follow-up correction is no longer likely), or
#   * it is older than SETTLE_HOURS (so a session nobody closes still drains), or
#   * its session was closed with the optional `end-session`.
AUTO = os.environ.get("MEMORYKIT_AUTO", "1") == "1"
SETTLE_TURNS = int(os.environ.get("MEMORYKIT_SETTLE_TURNS", "8"))
AUTO_DISTILLER = os.environ.get("MEMORYKIT_DISTILLER", "")
# Output is paid for in tokens by whoever reads it. Defaults are terse.
VERBOSE = os.environ.get("MEMORYKIT_VERBOSE", "0") == "1"
SNIPPET_CHARS = int(os.environ.get("MEMORYKIT_SNIPPET_CHARS", "200"))
# A raw turn retrieved this many times before it settles is distilled even
# when its phrasing scored under the salience floor: use is evidence.
SLEEP_MIN_HITS = int(os.environ.get("MEMORYKIT_SLEEP_MIN_HITS", "2"))
# Half-lives in days, by entry type (low-salience turns decay fastest).
HALF_LIFE = {"fact": 180, "note": 30, "rule": 365}
LOW_SALIENCE_HALF_LIFE = 7

IDENT_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?\b"
    r"|\b[0-9a-f]{16,}\b|\b\d+\b")


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------- markdown log
#
# Every capture and every state change (supersede, quarantine, restore,
# feedback, promote) is appended to a dated markdown log. Each line is human
# readable and carries one machine-readable HTML comment, so `reindex` can
# rebuild the whole database from markdown alone. The database is therefore
# genuinely disposable.

_REPLAYING = False
MK_RE = re.compile(r"<!-- mk:(\{.*\}) -->\s*$")


def md_dir_for(db_path):
    return os.environ.get("MEMORYKIT_MD_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(db_path)), "memory-log")


def log_md(db_path, record, display):
    """Append one record to today's markdown log (no-op during reindex)."""
    if _REPLAYING:
        return
    d = md_dir_for(db_path)
    os.makedirs(d, exist_ok=True)
    record.setdefault("at", now_iso())
    day = record["at"][:10]
    path = os.path.join(d, f"{day}.md")
    new = not os.path.exists(path)
    one_line = " ".join(display.split())[:300]
    with open(path, "a", encoding="utf-8") as fh:
        if new:
            fh.write(f"# memorykit log {day}\n\n")
        fh.write(f"- {one_line} <!-- mk:{json.dumps(record, ensure_ascii=False)} -->\n")


_MIGRATED = set()


def connect(db_path):
    db = sqlite3.connect(db_path, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        # v5: WAL so a reader never blocks the per-turn writer, FULL sync so a
        # committed turn survives power loss (the capture journal covers the
        # rest). Persistent per file; harmless if the file is read-only.
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
    except sqlite3.OperationalError:
        pass
    key = os.path.abspath(db_path)
    if key not in _MIGRATED:
        try:
            migrate(db)
            db.commit()
        except sqlite3.OperationalError:
            pass  # read-only or not initialized: nothing to migrate
        _MIGRATED.add(key)
    return db


def state_get(db, key, default=None):
    db.execute("CREATE TABLE IF NOT EXISTS memorykit_state "
               "(key TEXT PRIMARY KEY, value TEXT)")
    r = db.execute("SELECT value FROM memorykit_state WHERE key=?",
                   (key,)).fetchone()
    return json.loads(r[0]) if r else default


def state_set(db, key, value):
    db.execute("CREATE TABLE IF NOT EXISTS memorykit_state "
               "(key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT OR REPLACE INTO memorykit_state VALUES (?, ?)",
               (key, json.dumps(value)))


# The schema is embedded so the engine works from any install location
# (a copy in /usr/local/bin, a pip install, a zipapp). schema.sql ships beside
# it as documentation; a test keeps the two byte-identical.
SCHEMA = """-- memorykit schema — the vector-memory map index.
-- Markdown is the source of truth; this database is a disposable map.
-- Delete it any time and rebuild from markdown (see README: rebuilding).
-- One table for entries, one for derivation edges, one for feedback.
-- `status = 'rejected'` is the reversible quarantine (soft-delete).
-- Automation NEVER hard-deletes rows; eviction is a human decision.

CREATE TABLE IF NOT EXISTS memory_item (
    id               TEXT PRIMARY KEY NOT NULL,
    created          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    project          TEXT NOT NULL DEFAULT 'turns',
    session_key      TEXT,
    type             TEXT NOT NULL,              -- 'fact' | 'note' | 'rule'
    content          TEXT NOT NULL,              -- stored text, provenance header included
    salience         INTEGER NOT NULL DEFAULT 5, -- 1-10 importance
    confidence       REAL NOT NULL DEFAULT 1.0,  -- 0.0-1.0, see DESIGN.md
    metadata         TEXT,                       -- JSON: provenance, supersedes, keys, parents
    embedding        TEXT,                       -- JSON list of floats, or NULL
    status           TEXT NOT NULL DEFAULT 'confirmed', -- 'confirmed' | 'rejected'
    superseded_by    TEXT,                       -- id of the entry that replaced this one
    valid_from       TEXT NOT NULL,
    valid_to         TEXT,
    learned_at       TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_item_project ON memory_item (project);
CREATE INDEX IF NOT EXISTS idx_item_status ON memory_item (status);
CREATE INDEX IF NOT EXISTS idx_item_superseded ON memory_item (superseded_by);
CREATE INDEX IF NOT EXISTS idx_item_accessed ON memory_item (last_accessed_at);

-- Causal DAG: subject was derived from object (used by cascade pruning).
CREATE TABLE IF NOT EXISTS memory_edge (
    id         TEXT PRIMARY KEY NOT NULL,
    created    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    project    TEXT NOT NULL DEFAULT 'turns',
    subject_id TEXT NOT NULL,
    predicate  TEXT NOT NULL,              -- system: derived_from, distilled_from;
                                           -- user/agent (via `relate`): any other word
    object_id  TEXT NOT NULL,
    source_id  TEXT,
    weight     REAL NOT NULL DEFAULT 1.0   -- edge strength, set by `relate`
);

CREATE INDEX IF NOT EXISTS idx_edge_object ON memory_edge (object_id, predicate);
CREATE INDEX IF NOT EXISTS idx_edge_subject ON memory_edge (subject_id, predicate);

-- Explicit retrieval feedback: does this entry actually help?
CREATE TABLE IF NOT EXISTS memory_feedback (
    id          TEXT PRIMARY KEY NOT NULL,
    created     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    memory_id   TEXT NOT NULL REFERENCES memory_item(id) ON DELETE CASCADE,
    rating      TEXT NOT NULL,              -- '+1' | '-1'
    comment     TEXT,
    FOREIGN KEY (memory_id) REFERENCES memory_item(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback (memory_id);
"""

SYSTEM_PREDICATES = ("derived_from", "distilled_from")


def migrate(db):
    """Bring an older database up to the current schema. Additive only."""
    cols = [r[1] for r in db.execute("PRAGMA table_info(memory_edge)")]
    if cols and "weight" not in cols:
        db.execute("ALTER TABLE memory_edge ADD COLUMN weight REAL NOT NULL DEFAULT 1.0")
    if cols:
        db.execute("CREATE INDEX IF NOT EXISTS idx_edge_subject"
                   " ON memory_edge (subject_id, predicate)")


def init_db(db_path):
    d = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(d, exist_ok=True)
    db = connect(db_path)
    db.executescript(SCHEMA)
    migrate(db)
    db.commit()
    db.close()
    print(f"initialized {db_path}")


# ---------------------------------------------------------------- embeddings

_embedder = None
_embedder_tried = False


def get_embedder():
    """Local embedder via fastembed, or None when unavailable (never fatal)."""
    global _embedder, _embedder_tried
    if _embedder_tried:
        return _embedder
    _embedder_tried = True
    if os.environ.get("MEMORYKIT_NO_EMBED") == "1":
        _embedder = None  # forced keyword mode (tests, low-resource hosts)
        return None
    try:
        from fastembed import TextEmbedding
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            _embedder = TextEmbedding(model_name=MODEL, cache_dir=MODEL_DIR)
        except TypeError:  # very old fastembed without cache_dir
            _embedder = TextEmbedding(model_name=MODEL)
    except Exception:
        _embedder = None
    return _embedder


def embed_texts(texts):
    """Return a list of float lists, or None when no embedder is installed."""
    emb = get_embedder()
    if emb is None:
        return None
    return [list(map(float, v)) for v in emb.embed(texts)]


def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def load_vec(row):
    try:
        return json.loads(row["embedding"]) if row["embedding"] else None
    except (json.JSONDecodeError, TypeError):
        return None


def live_rows(db, project=None):
    q = ("SELECT * FROM memory_item WHERE status != 'rejected' "
         "AND superseded_by IS NULL")
    args = []
    if project:
        q += " AND project = ?"
        args.append(project)
    return db.execute(q, args).fetchall()


# ---------------------------------------------------------------- salience

def salience_heuristic(text):
    s = 3
    if DECISION_RE.search(text):
        s += 2
    if CORRECTION_RE.search(text):
        s += 2
    if text.rstrip().endswith("?"):
        s += 1
    if len(text) > 300:
        s += 1
    if EMOTION_RE.search(text):
        s += 1
    return max(1, min(10, s))


_NEGWORDS = {"not", "never", "no", "isn", "aren", "wasn", "don", "doesn",
             "didn", "won", "can", "cannot", "without", "t"}


def value_change(a, b):
    """True when two texts differ ONLY in numbers or negation words: the
    same statement with a changed value ("costs 40" vs "costs 55"). That is
    an update or a contradiction, never a restatement."""
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    diff = ta ^ tb
    return bool(diff) and all(t.isdigit() or t in _NEGWORDS for t in diff)


def novelty_adjust(db, text, salience):
    """Restatements can't inflate into importance; novelty gets +1.

    Cosine against every live entry, plus a difflib check (which catches
    near-verbatim restatements that mean-pooling under-scores when lengths
    differ) on the nearest candidates by cosine and by BM25. Never raises: on any
    failure the heuristic salience stands.
    """
    try:
        rows = live_rows(db)
        if not rows:
            return min(10, salience + 1), "novel"
        vecs = embed_texts([text[:4000]])
        vec = vecs[0] if vecs else None
        # Score cosine for all rows in one matrix product, then run the
        # (slow) difflib check only on the 20 nearest by cosine plus the 20
        # nearest by BM25. BM25 candidates keep keyword mode honest: with no
        # embedder every cosine is 0, and cosine-only candidates would just
        # be the oldest rows.
        sims = _doc_sims(db, rows, vec) if vec else [0.0] * len(rows)
        lex, _ = _bm25(_tokens(text), _doc_tokens(db, rows))
        order = lambda xs: sorted(range(len(rows)), key=lambda i: -xs[i])[:20]
        cand = set(order(sims)) | set(i for i in order(lex) if lex[i] > 0)
        scored = []
        for i in cand:
            body = _body(rows[i]["content"])
            if value_change(text, body):
                continue  # a changed value is news, not a restatement
            sm = difflib.SequenceMatcher(None, text[:2000], body[:2000])
            t = sm.ratio() if sm.real_quick_ratio() > NOVELTY_RESTATEMENT_TXT else 0.0
            scored.append((max(sims[i], 0.0), t))
        best_cos = max(c for c, _ in scored) if scored else 0.0
        best_txt = max(t for _, t in scored) if scored else 0.0
        if best_cos > NOVELTY_RESTATEMENT_COS or best_txt > NOVELTY_RESTATEMENT_TXT:
            return min(salience, 3), \
                f"restatement(cos={best_cos:.3f},txt={best_txt:.3f})"
        if vec and best_cos < NOVELTY_NOVEL_COS:
            return min(10, salience + 1), f"novel(cos={best_cos:.3f})"
        return salience, f"related(cos={best_cos:.3f})"
    except Exception:
        return salience, "unknown"


# ---------------------------------------------------------------- tombstoning

def is_exempt(row, md):
    return row["salience"] >= 9 or (isinstance(md, dict) and md.get("protected"))


def tombstone(db, eid, reason, cascade=False, at=None):
    """Reversible soft-delete + recursive cascade prune of derived children.

    Exempt entries (salience >= 9 or metadata.protected) are never pulled
    down by a cascade; they are flagged `orphaned_parent` instead."""
    row = db.execute(
        "SELECT status, metadata, salience FROM memory_item WHERE id = ?",
        (eid,)).fetchone()
    if not row or row["status"] == "rejected":
        return
    try:
        md = json.loads(row["metadata"]) if row["metadata"] else {}
    except json.JSONDecodeError:
        md = {}
    if not isinstance(md, dict):
        md = {}
    if cascade and is_exempt(row, md):
        md["orphaned_parent"] = reason
        db.execute("UPDATE memory_item SET metadata=? WHERE id=?",
                   (json.dumps(md), eid))
        return
    at = at or now_iso()
    md["tombstoned_at"] = at
    md["tombstone_reason"] = reason
    db.execute(
        "UPDATE memory_item SET status='rejected', metadata=?, updated=? "
        "WHERE id=?", (json.dumps(md), at, eid))
    children = db.execute(
        "SELECT subject_id FROM memory_edge "
        "WHERE object_id=? AND predicate='derived_from'", (eid,)).fetchall()
    for ch in children:
        tombstone(db, ch["subject_id"], f"cascade from {eid}", cascade=True,
                  at=at)


# ---------------------------------------------------------------- commands

def cmd_capture(args):
    db = connect(args.db)
    text = args.text.strip()
    if len(text) < 10:
        print("SKIP: turn too short to index", file=sys.stderr)
        return 0
    if args.check:
        hits = search_rows(db, text, 5)
        # Advisory only; nothing is auto-resolved. Terse: id, type, snippet.
        for r, score, sim in hits:
            print(f"related {short(r['id'])} {r['type']} "
                  f"{snippet(r['content'], text, SNIPPET_CHARS // 2)}")
    provenance = args.provenance or SPEAKER_PROVENANCE[args.speaker]
    if args.speaker == "agent" and provenance not in ("agent-said",
                                                      "agent-inferred"):
        # The firewall is enforced, not advisory: an agent cannot stamp its
        # own words as user-said or tool-observed.
        print(f"REFUSED: speaker=agent cannot claim provenance={provenance}",
              file=sys.stderr)
        return 2
    mtype, confidence = PROVENANCE[provenance]
    salience = args.salience if args.salience else salience_heuristic(text)
    novelty = "manual" if args.salience else "unknown"
    if not args.salience:
        salience, novelty = novelty_adjust(db, text, salience)
    vecs = embed_texts([text[:4000]])
    metadata = {
        "provenance": provenance,
        "captured_at": now_iso(),
        "turn": args.turn,
        "salience_reason": "manual" if args.salience else "heuristic",
        "novelty": novelty,
    }
    # Attribution: who said it, kept apart from who it quotes. Quoted text
    # lives in metadata, not content, so it is never indexed as the
    # quoter's own words (search finds it under its original speaker).
    src = (getattr(args, "source", "") or "").strip()
    if src:
        metadata["source"] = src
    qtext = (getattr(args, "quote", "") or "").strip()
    if qtext:
        metadata["quote"] = {"source": (getattr(args, "quote_source", "") or "").strip(),
                             "text": qtext[:500]}
    if args.supersedes:
        metadata["supersedes"] = [s.strip() for s in args.supersedes.split(",") if s.strip()]
    if args.supersession_key:
        metadata["supersession_key"] = args.supersession_key.strip()
    parent_ids = [p.strip() for p in (args.parent_ids or "").split(",") if p.strip()]
    if parent_ids:
        metadata["parent_ids"] = parent_ids
    content = f"[turn provenance={provenance}]\n{text}"
    new_id = getattr(args, "force_id", None) or uuid.uuid4().hex[:24]
    if getattr(args, "force_id", None) and db.execute(
            "SELECT 1 FROM memory_item WHERE id=?", (new_id,)).fetchone():
        db.close()  # idempotent: a replayed or retried turn is already stored
        print(f"exists id={short(new_id)}")
        return 0
    now = getattr(args, "force_at", None) or now_iso()
    if args.protected:
        metadata["protected"] = True
    db.execute(
        """INSERT INTO memory_item
           (id, created, updated, project, session_key, type, content,
            salience, confidence, metadata, embedding, status,
            valid_from, learned_at, last_accessed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed', ?, ?, ?)""",
        (new_id, now, now, args.project, args.session or "", mtype, content,
         salience, confidence, json.dumps(metadata),
         json.dumps(vecs[0]) if vecs else None, now, now, now))
    retired = []
    targets = list(metadata.get("supersedes", []))
    for old_id in list(targets):
        # A correction aimed at a raw turn that sleep already distilled must
        # also retire the semantic entry that now carries that fact.
        r = db.execute("SELECT metadata FROM memory_item WHERE id=?",
                       (old_id,)).fetchone()
        try:
            into = json.loads(r["metadata"] or "{}").get("distilled_into") if r else None
        except json.JSONDecodeError:
            into = None
        if into and into not in targets:
            targets.append(into)
    for old_id in targets:
        db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?",
                   (new_id, old_id))
        tombstone(db, old_id, f"superseded by {new_id}", at=now)
        retired.append(old_id)
    skey = metadata.get("supersession_key")
    if skey:
        prior = db.execute(
            """SELECT id FROM memory_item WHERE id != ? AND status != 'rejected'
               AND json_extract(metadata, '$.supersession_key') = ?""",
            (new_id, skey)).fetchall()
        for p in prior:
            db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?",
                       (new_id, p["id"]))
            tombstone(db, p["id"], f"supersession key {skey} -> {new_id}",
                      at=now)
            retired.append(p["id"])
        if prior:
            print(f"supersession-key {skey}: retired {len(prior)} prior version(s)")
    for parent in parent_ids:
        db.execute(
            """INSERT INTO memory_edge
               (id, created, updated, project, subject_id, predicate,
                object_id, source_id)
               VALUES (?, ?, ?, ?, ?, 'derived_from', ?, ?)""",
            (uuid.uuid4().hex[:24], now, now, args.project, new_id, parent, new_id))
    db.commit()
    db.close()
    _crash("after_commit")
    if not getattr(args, "no_log", False):
        log_md(args.db, {
            "op": "capture", "id": new_id, "at": now, "speaker": args.speaker,
            "provenance": provenance, "text": text, "salience": salience,
            "salience_manual": bool(args.salience), "project": args.project,
            "session": args.session or "", "turn": args.turn,
            "supersedes": sorted(set(retired)), "supersession_key": skey or "",
            "parents": parent_ids, "protected": bool(args.protected),
            "novelty": novelty, "source": src,
            "quote": metadata.get("quote", {})},
            f"[{provenance}] {text}")
    # Terse receipt: the id and the one number that decides its fate.
    print(f"saved id={short(new_id)} salience={salience}"
          + (f" provenance={provenance}" if VERBOSE else ""))
    if AUTO and not _REPLAYING and not getattr(args, "no_auto", False):
        upkeep_step(args.db)
    if vecs is None and not os.environ.get("MEMORYKIT_NO_EMBED"):
        db = connect(args.db)
        if not state_get(db, "warned_no_embed", False):
            print("note: no embedder installed; keyword mode. Run `backfill` "
                  "after `pip install fastembed` (shown once)", file=sys.stderr)
            state_set(db, "warned_no_embed", True)
            db.commit()
        db.close()
    return 0


_TOK = re.compile(r"\w+")


# Function words that appear in nearly every entry. They carry no lexical
# evidence and only dilute BM25. Negations are deliberately NOT here:
# "not", "never", "no", "without" change meaning, and value_change() and
# the conflict flags depend on them.
STOPWORDS = frozenset("""
a an the and or but if then so of to in on at by for from with as into onto
over under about after before between through during is am are was were be
been being do does did doing have has had having it its this that these those
there here i me my mine we us our you your he him his she her they them their
what which who whom whose when where why how all any each both few more most
other some such only own same than too very can will just should would could
may might must shall also up down out off again once s t ll d re ve m
""".split())


def _tokens(text):
    """Lexical tokens: lowercase words, 2+ chars, stopwords removed. Used for
    both queries and documents, so they always agree."""
    return [t for t in _TOK.findall((text or "").lower())
            if len(t) > 1 and t not in STOPWORDS]


def _bm25(query_toks, docs, k1=1.2, b=0.75):
    """Plain BM25 over the candidate set. Stdlib only. One pass: only the
    query's terms are counted, and only documents containing one are scored."""
    n = len(docs)
    if not n:
        return [], {}
    q = set(query_toks)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = dict.fromkeys(q, 0)
    hits = []
    for i, d in enumerate(docs):
        common = q.intersection(d)
        if common:
            for t in common:
                df[t] += 1
            hits.append((i, common))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q}
    out = [0.0] * n
    for i, common in hits:
        d = docs[i]
        norm = k1 * (1 - b + b * len(d) / avgdl)
        sc = 0.0
        for t in common:
            f = d.count(t)
            sc += idf[t] * f * (k1 + 1) / (f + norm)
        out[i] = sc
    return out, idf


def _cache_tables(db):
    db.execute("CREATE TABLE IF NOT EXISTS lex_cache (id TEXT PRIMARY KEY, toks TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS vec_cache (id TEXT PRIMARY KEY, v BLOB)")


def _doc_tokens(db, rows):
    """Per-row lexical tokens, cached by id (entry text never changes after
    capture). Derived data: safe to drop, refilled on demand, never logged."""
    _cache_tables(db)
    have = dict(db.execute("SELECT id, toks FROM lex_cache"))
    out, new = [], []
    for r in rows:
        t = have.get(r["id"])
        if t is None:
            t = " ".join(_tokens(_body(r["content"])))
            new.append((r["id"], t))
        out.append(t.split() if t else [])
    if new:
        db.executemany("INSERT OR REPLACE INTO lex_cache VALUES (?, ?)", new)
        db.commit()
    return out


def _doc_sims(db, rows, qvec):
    """Cosine of every row to the query. With numpy (always present when the
    embedder is), vectors are cached as normalized float32 blobs and scored
    in one matrix product; otherwise falls back to pure Python."""
    try:
        import numpy as np
    except ImportError:
        return [cosine(qvec, v) if (v := load_vec(r)) else 0.0 for r in rows]
    _cache_tables(db)
    have = dict(db.execute("SELECT id, v FROM vec_cache"))
    dim = len(qvec)
    mat = np.zeros((len(rows), dim), dtype=np.float32)
    new = []
    for i, r in enumerate(rows):
        blob = have.get(r["id"])
        if blob is None:
            v = load_vec(r)
            if not v or len(v) != dim:
                continue
            a = np.asarray(v, dtype=np.float32)
            nrm = float(np.linalg.norm(a))
            if not nrm:
                continue
            a /= nrm
            new.append((r["id"], a.tobytes()))
            mat[i] = a
        elif len(blob) == dim * 4:
            mat[i] = np.frombuffer(blob, dtype=np.float32)
    if new:
        db.executemany("INSERT OR REPLACE INTO vec_cache VALUES (?, ?)", new)
        db.commit()
    q = np.asarray(qvec, dtype=np.float32)
    q /= (np.linalg.norm(q) or 1.0)
    return (mat @ q).tolist()


def _minmax(xs):
    """Scale to 0..1 across candidates. A tie means no discrimination, not
    no match: tied positive signals score 1.0 (a one-entry store must still
    find its entry)."""
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return [1.0 if x > 0 else 0.0 for x in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _source_of(r):
    try:
        return json.loads(r["metadata"] or "{}").get("source", "")
    except (json.JSONDecodeError, TypeError):
        return ""


def search_rows(db, query, limit, source=""):
    """Hybrid retrieval, live rows only.

    relevance = (1-w) * minmax(cosine) + w * minmax(BM25), where w is how much
    of the query's IDF mass the best lexical match covers, squared. Exact
    wording leans lexical, paraphrase leans semantic. Keyword mode is BM25.
    Salience and recency only TILT relevance (x0.8..1.0 and x0.9..1.0);
    they never outrank it. On a 36-query benchmark the old product
    (sim x salience) put the target in the top 5 about 10% of the time."""
    rows = live_rows(db)
    if source:
        want = source.strip().lower()
        rows = [r for r in rows if _source_of(r).lower() == want]
    if not rows:
        return []
    vecs = embed_texts([query[:4000]])
    qvec = vecs[0] if vecs else None
    qt = _tokens(query)
    docs = _doc_tokens(db, rows)
    lex, idf = _bm25(qt, docs)
    best = max(range(len(rows)), key=lambda i: lex[i])
    tot = sum(idf.values()) or 1.0
    cover = sum(v for t, v in idf.items() if t in set(docs[best])) / tot
    w = cover ** 2
    sims = None
    if qvec:
        sims = _doc_sims(db, rows, qvec)
    if sims and any(sims):
        sn, ln = _minmax(sims), _minmax(lex) if max(lex) > 0 else [0.0] * len(lex)
        rel = [(1 - w) * a + w * b for a, b in zip(sn, ln)]
    else:
        # Keyword mode: rank by BM25; report query-token coverage as the
        # absolute "sim" (callers like correlate threshold on it).
        qs = set(t for t in qt if len(t) > 2)
        sims = [len(qs & set(d)) / len(qs) if qs else 0.0 for d in docs]
        rel = lex
    now = datetime.datetime.now(datetime.timezone.utc)
    scored = []
    for r, rv, lx, sm in zip(rows, rel, lex, sims):
        # Filter on the raw signals, never on the normalized one: min-max puts
        # the weakest candidate at 0, which would make it unreachable in a
        # small store. A small raw-similarity term keeps it ordered.
        if lx <= 0 and sm <= 0:
            continue
        rv = rv + 1e-3 * max(sm, 0.0)
        try:
            accessed = datetime.datetime.fromisoformat(r["last_accessed_at"])
            age_days = max(0.0, (now - accessed).total_seconds() / 86400)
        except (ValueError, TypeError):
            age_days = 0.0
        recency = 1.0 / (1.0 + age_days / 30.0)
        score = rv * (0.8 + 0.2 * r["salience"] / 10.0) * (0.9 + 0.1 * recency)
        scored.append((r, score, sm))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def snippet(text, query, width, qvec=None):
    """The `width`-char window of `text` that best answers `query`.

    Windows are scored on query-word overlap (5-char stems, so "stored"
    meets "store"); when no window shares a word (pure paraphrase) and an
    embedder is present, windows are ranked by cosine instead. Costs CPU,
    never tokens; saves the reader from paying for a whole entry."""
    text = " ".join(_body(text).split())
    if len(text) <= width:
        return text
    stem = lambda w: w[:5]
    qs = {stem(t) for t in _tokens(query) if len(t) > 2}
    words = text.split(" ")
    pos, i = [], 0
    for wd in words:
        pos.append(i)
        i += len(wd) + 1
    hitw = [{stem(t) for t in _tokens(wd)} & qs for wd in words]
    starts, scores = [], []
    step = max(1, len(words) // 60)  # cap the number of candidate windows
    # Candidate windows open just before a matched word, so ties resolve to
    # a window that starts at the evidence instead of trailing into it.
    cands = sorted({max(0, k - 2) for k, h in enumerate(hitw) if h}) or \
        list(range(0, len(words), step))
    for k in cands:
        j, got = k, set()
        while j < len(words) and pos[j] - pos[k] < width:
            got |= hitw[j]  # distinct query words covered, not repeats
            j += 1
        starts.append(k)
        scores.append(len(got))
    best_k = starts[max(range(len(starts)), key=lambda x: scores[x])]
    if max(scores) == 0 and qvec:
        wins = [text[pos[k]:pos[k] + width] for k in starts]
        vecs = embed_texts(wins)
        if vecs:
            best_k = starts[max(range(len(wins)), key=lambda x: cosine(qvec, vecs[x]))]
    start = pos[best_k]
    end = start + width
    out = text[start:end].rsplit(" ", 1)[0] if end < len(text) else text[start:]
    return ("…" if start else "") + out + ("…" if end < len(text) else "")


def search_lines(db, query, limit, chars=SNIPPET_CHARS, budget=0, source="",
                 verbose=False, skip_ids=()):
    """Ranked hits as terse lines within an approximate token budget
    (4 chars a token). Bumps access stats for what it shows."""
    hits = search_rows(db, query, limit + len(skip_ids), source)
    qv = embed_texts([query[:4000]]) if hits else None
    qv = qv[0] if qv else None
    used, lines, ids, seen = 0, [], [], set()
    for r, score, sim in hits:
        if r["id"] in skip_ids or len(lines) >= limit:
            continue
        snip = snippet(r["content"], query, chars, qv)
        if snip in seen:
            continue  # identical text already shown: pay for it once
        seen.add(snip)
        if verbose:
            line = (f"- id={r['id']} type={r['type']} sal={r['salience']} "
                    f"conf={r['confidence']} score={score:.3f}\n  {snip}")
        else:
            src = _source_of(r)
            line = f"{short(r['id'])} {r['type']}{'@' + src if src else ''} {snip}"
        cost = len(line) // 4 + 1
        if budget and lines and used + cost > budget:
            break
        lines.append(line)
        ids.append(r["id"])
        used += cost
        db.execute("UPDATE memory_item SET last_accessed_at=? WHERE id=?",
                   (now_iso(), r["id"]))
        if r["project"] != SEMANTIC_PROJECT:
            md = json.loads(r["metadata"] or "{}")
            md["hits"] = md.get("hits", 0) + 1
            db.execute("UPDATE memory_item SET metadata=? WHERE id=?",
                       (json.dumps(md), r["id"]))
    db.commit()
    return lines, ids


def cmd_search(args):
    db = connect(args.db)
    lines, _ = search_lines(db, args.query, args.limit, args.chars,
                            getattr(args, "budget", 0) or 0,
                            getattr(args, "source", ""), args.verbose)
    db.close()
    print("\n".join(lines) if lines else "(no matches)")


def cmd_get(args):
    db = connect(args.db)
    r = db.execute("SELECT * FROM memory_item WHERE id=?", (args.id,)).fetchone()
    db.close()
    if not r:
        print(f"no entry {args.id}")
        return 1
    if getattr(args, "text", False):
        print(_body(r["content"]))  # just the words: cheapest full read
        return 0
    d = dict(r)
    if not getattr(args, "with_embedding", False):
        d.pop("embedding", None)  # 384 floats is thousands of tokens of noise
    print(json.dumps(d, default=str, ensure_ascii=False,
                     indent=2 if getattr(args, "pretty", False) else None))
    return 0


def cmd_supersede(args):
    db = connect(args.db)
    db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?",
               (args.new, args.old))
    at = now_iso()
    tombstone(db, args.old, f"superseded by {args.new}", at=at)
    db.commit()
    db.close()
    log_md(args.db, {"op": "supersede", "old": args.old, "new": args.new,
                     "at": at},
           f"supersede {args.old} -> {args.new}")
    print(f"{args.old} superseded by {args.new} (old row kept, filtered from search)")


def identifier_only_diff(a, b):
    """True when two texts differ ONLY in identifier-like tokens."""
    if a == b:
        return False
    return IDENT_RE.sub("", a).strip() == IDENT_RE.sub("", b).strip()


def provenance_rank(row):
    # Facts outrank notes; higher confidence breaks ties.
    return (0 if row["type"] == "fact" else 1, -row["confidence"])


def _norm_text(c):
    c = re.sub(r"^\[turn provenance=[^\]]*\]\n", "", c or "")
    return " ".join(c.lower().split())


def dedup_candidates(rows):
    """Pairs (a, b, cos, txt) that pass every merge gate.

    Blocking: cosine is computed first (vectorized with numpy when present,
    in row chunks), and difflib only runs on pairs that clear the cosine
    gate. Rows without vectors (keyword mode) merge only on an exact match
    of normalized text, since there is no meaning signal to gate on.
    """
    out = []
    with_vec = [(r, load_vec(r)) for r in rows]
    vec_rows = [(r, v) for r, v in with_vec if v]
    no_vec = [r for r, v in with_vec if not v]
    pairs = []
    if vec_rows:
        try:
            import numpy as np
            M = np.array([v for _, v in vec_rows], dtype=np.float32)
            M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
            step = 512
            for i0 in range(0, len(M), step):
                sims = M[i0:i0 + step] @ M.T
                ii, jj = np.nonzero(sims >= DEDUP_COSINE)
                for i, j in zip(ii.tolist(), jj.tolist()):
                    gi = i0 + i
                    if j > gi:
                        pairs.append((gi, j, float(sims[i, j])))
        except ImportError:
            for gi in range(len(vec_rows)):
                for j in range(gi + 1, len(vec_rows)):
                    c = cosine(vec_rows[gi][1], vec_rows[j][1])
                    if c >= DEDUP_COSINE:
                        pairs.append((gi, j, c))
    for gi, j, c in pairs:
        a, b = vec_rows[gi][0], vec_rows[j][0]
        if a["type"] != b["type"]:
            continue
        txt = difflib.SequenceMatcher(None, (a["content"] or "")[:4000],
                                      (b["content"] or "")[:4000]).ratio()
        if txt >= DEDUP_TEXT and not identifier_only_diff(
                a["content"] or "", b["content"] or ""):
            out.append((a, b, c, txt))
    buckets = {}
    for r in no_vec:
        buckets.setdefault((r["type"], _norm_text(r["content"])), []).append(r)
    for group in buckets.values():
        for b in group[1:]:
            out.append((group[0], b, 0.0, 1.0))
    return out


def cmd_dedup(args):
    db = connect(args.db)
    rows = live_rows(db)
    merges = []
    seen = set()
    dedup_at = now_iso()
    for a, b, cos, txt in dedup_candidates(rows):
        if a["id"] in seen or b["id"] in seen:
            continue
        keep, drop = sorted([a, b], key=provenance_rank)
        merges.append((keep["id"], drop["id"], cos, txt))
        seen.add(drop["id"])
    if not merges:
        print("no merge candidates")
        db.close()
        return 0
    for keep, drop, cos, txt in merges:
        print(f"merge: keep {keep} drop {drop} (cos={cos:.3f} txt={txt:.3f})")
        if args.apply:
            db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?",
                       (keep, drop))
            tombstone(db, drop, f"dedup merged into {keep}", at=dedup_at)
    if args.apply:
        db.commit()
        for keep, drop, _, _ in merges:
            log_md(args.db, {"op": "supersede", "old": drop, "new": keep,
                             "reason": "dedup", "at": dedup_at},
                   f"dedup {drop} -> {keep}")
        print(f"applied {len(merges)} merge(s); dropped rows kept as rejected")
    else:
        print("dry run — pass --apply to merge")
    db.close()
    return 0


def restore(db, eid, at=None):
    """Undo a quarantine and every cascade it caused. Returns rows restored."""
    # A restore is a human judgment that the entry is worth keeping, so it
    # restarts the decay clock (restored_at). learned_at keeps the true age.
    at = at or now_iso()
    row = db.execute("SELECT metadata FROM memory_item WHERE id=?",
                     (eid,)).fetchone()
    try:
        md = json.loads(row["metadata"]) if row and row["metadata"] else {}
    except json.JSONDecodeError:
        md = {}
    if not isinstance(md, dict):
        md = {}
    md["restored_at"] = at
    md.pop("tombstoned_at", None)
    db.execute("UPDATE memory_item SET status='confirmed', updated=?, metadata=?"
               " WHERE id=?", (at, json.dumps(md), eid))
    n = 1
    kids = db.execute(
        "SELECT subject_id FROM memory_edge WHERE object_id=? "
        "AND predicate='derived_from'", (eid,)).fetchall()
    for k in kids:
        r = db.execute("SELECT status, metadata FROM memory_item WHERE id=?",
                       (k["subject_id"],)).fetchone()
        if r and r["status"] == "rejected" and \
                f"cascade from {eid}" in (r["metadata"] or ""):
            n += restore(db, k["subject_id"], at)
    return n


def gc_score(row, feedback):
    """S = U x e^(-lambda x dt) x R. See DESIGN.md for the terms."""
    u = row["confidence"] * (row["salience"] / 10.0)
    fb = feedback.get(row["id"], (0, 0))
    u *= min(3.0, 1.5 ** fb[0]) if fb[0] else 1.0
    u *= max(0.25, 0.5 ** fb[1]) if fb[1] else 1.0
    nowdt = datetime.datetime.now(datetime.timezone.utc)

    def days_since(col):
        try:
            return max(0.0, (nowdt - datetime.datetime.fromisoformat(
                row[col])).total_seconds() / 86400)
        except (ValueError, TypeError):
            return 0.0
    # Decay runs from when the entry was learned. Access only feeds the
    # bounded R term; letting a search reset decay would make retrieval
    # itself the survival signal (echo reinforcement, failure mode 2).
    age = days_since("learned_at")
    try:
        ra = (json.loads(row["metadata"] or "{}") or {}).get("restored_at")
        if ra:
            age = min(age, max(0.0, (nowdt - datetime.datetime.fromisoformat(
                ra)).total_seconds() / 86400))
    except (ValueError, TypeError, AttributeError):
        pass
    since_access = days_since("last_accessed_at")
    half = LOW_SALIENCE_HALF_LIFE if row["salience"] <= 3 else \
        HALF_LIFE.get(row["type"], 30)
    lam = math.log(2) / half
    r = 1.0 + 0.5 * math.exp(-since_access / 30.0)
    return u * math.exp(-lam * age) * r


def cmd_gc(args):
    db = connect(args.db)
    if args.restore:
        at = now_iso()
        restored = restore(db, args.restore, at)
        db.commit()
        log_md(args.db, {"op": "restore", "id": args.restore, "at": at},
               f"restore {args.restore}")
        print(f"restored {args.restore} (+{restored - 1} cascaded child(ren))")
        db.close()
        return 0
    if args.eviction_report:
        cutoff = (datetime.datetime.now(datetime.timezone.utc)
                  - datetime.timedelta(days=GC_QUARANTINE_DAYS)).isoformat()
        rows = db.execute(
            "SELECT id, type, salience, metadata FROM memory_item "
            "WHERE status='rejected' AND updated < ?", (cutoff,)).fetchall()
        print(f"{len(rows)} quarantined entr(ies) past {GC_QUARANTINE_DAYS} days "
              "(review manually; this tool never hard-deletes):")
        for r in rows:
            print(f"- {r['id']} type={r['type']} sal={r['salience']}")
        db.close()
        return 0
    feedback = {}
    for f in db.execute(
            "SELECT memory_id, rating, COUNT(*) c FROM memory_feedback "
            "GROUP BY memory_id, rating").fetchall():
        pos, neg = feedback.get(f["memory_id"], (0, 0))
        if f["rating"] == "+1":
            pos = f["c"]
        elif f["rating"] == "-1":
            neg = f["c"]
        feedback[f["memory_id"]] = (pos, neg)
    quarantined = []
    gc_at = now_iso()
    for row in live_rows(db):
        md = json.loads(row["metadata"]) if row["metadata"] else {}
        if is_exempt(row, md):
            continue  # exempt: high-importance rules and protected entries
        score = gc_score(row, feedback)
        if score < GC_NOISE_FLOOR:
            quarantined.append((row["id"], score))
            if args.apply:
                tombstone(db, row["id"], f"gc score {score:.4f} below floor",
                          at=gc_at)
    if args.apply:
        db.commit()
        for eid, score in quarantined:
            log_md(args.db, {"op": "quarantine", "id": eid, "at": gc_at,
                             "reason": f"gc score {score:.4f} below floor"},
                   f"quarantine {eid}")
        print(f"quarantined {len(quarantined)} entr(ies) (reversible)")
    else:
        print(f"dry run: {len(quarantined)} entr(ies) would quarantine")
    for eid, score in quarantined:
        print(f"- {eid} score={score:.4f}")
    db.close()
    return 0


def cmd_backfill(args):
    db = connect(args.db)
    rows = db.execute(
        "SELECT id, content FROM memory_item WHERE embedding IS NULL").fetchall()
    if not rows:
        print("nothing to backfill")
        return 0
    vecs = embed_texts([(r["content"] or "")[:4000] for r in rows])
    if vecs is None:
        print("no embedder installed — pip install fastembed, then retry",
              file=sys.stderr)
        return 1
    for r, v in zip(rows, vecs):
        db.execute("UPDATE memory_item SET embedding=? WHERE id=?",
                   (json.dumps(v), r["id"]))
    db.commit()
    print(f"backfilled {len(rows)} vector(s)")
    db.close()
    return 0


def cmd_correlate(args):
    """Approximate cross-layer audit: every substantive line of the curated
    markdown is checked against the vector index. Lines with no neighbor
    above threshold are ORPHAN candidates — reported, never auto-deleted.
    The curated layer outranks the index: an orphan may mean the index is
    stale, not that the claim is wrong."""
    db = connect(args.db)
    try:
        lines = open(args.markdown, encoding="utf-8").read().splitlines()
    except OSError as e:
        print(f"cannot read {args.markdown}: {e}", file=sys.stderr)
        return 1
    claims = [ln.strip(" -#*\t") for ln in lines
              if len(ln.strip()) > 40 and not ln.strip().startswith("<!--")]
    ok = orphans = 0
    for claim in claims:
        hits = search_rows(db, claim, 1)
        vec_mode = bool(hits) and load_vec(hits[0][0]) is not None \
            and get_embedder() is not None
        floor = CORRELATE_COSINE_MIN if vec_mode else CORRELATE_KEYWORD_MIN
        if hits and hits[0][2] > floor:
            ok += 1
        else:
            orphans += 1
            print(f"ORPHAN?: {claim[:160]}")
    print(f"checked {len(claims)} claims: {ok} ok, {orphans} orphan candidates")
    db.close()
    return 0


def cmd_feedback(args):
    db = connect(args.db)
    if not db.execute("SELECT 1 FROM memory_item WHERE id=?",
                      (args.id,)).fetchone():
        print(f"no entry {args.id}", file=sys.stderr)
        return 1
    at = getattr(args, "force_at", None) or now_iso()
    db.execute("INSERT INTO memory_feedback (id, created, memory_id, rating, comment)"
               " VALUES (?, ?, ?, ?, ?)",
               (uuid.uuid4().hex[:24], at, args.id, args.rating, args.comment))
    db.commit()
    db.close()
    log_md(args.db, {"op": "feedback", "id": args.id, "rating": args.rating,
                     "comment": args.comment or "", "at": at},
           f"feedback {args.rating} on {args.id}")
    print(f"recorded {args.rating} on {args.id}")
    return 0


def cmd_promote(args):
    """Promote a note to fact ONLY on independent evidence.

    The evidence must be a different, live entry with user-said or
    tool-observed provenance, and it must not be derived from the note
    (directly or transitively). Repetition never promotes."""
    db = connect(args.db)
    note = db.execute("SELECT * FROM memory_item WHERE id=?", (args.id,)).fetchone()
    ev = db.execute("SELECT * FROM memory_item WHERE id=?",
                    (args.evidence,)).fetchone()

    def fail(msg):
        print(f"REFUSED: {msg}", file=sys.stderr)
        db.close()
        return 2
    if not note or not ev:
        return fail("note or evidence not found")
    if note["id"] == ev["id"]:
        return fail("an entry cannot be its own evidence")
    if note["type"] != "note":
        return fail(f"entry is already type={note['type']}")
    if ev["status"] == "rejected" or ev["superseded_by"]:
        return fail("evidence is not live")
    ev_md = json.loads(ev["metadata"] or "{}")
    if ev_md.get("provenance") not in ("user-said", "tool-observed"):
        return fail("evidence must be user-said or tool-observed")
    frontier, seen = [ev["id"]], set()
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for e in db.execute("SELECT object_id FROM memory_edge WHERE subject_id=?"
                            " AND predicate='derived_from'", (cur,)).fetchall():
            if e["object_id"] == note["id"]:
                return fail("evidence is derived from the note itself")
            frontier.append(e["object_id"])
    md = json.loads(note["metadata"] or "{}")
    md["promoted_by"] = ev["id"]
    md["promoted_at"] = getattr(args, "force_at", None) or now_iso()
    db.execute("UPDATE memory_item SET type='fact', confidence=?, metadata=?, "
               "updated=? WHERE id=?",
               (max(note["confidence"], ev["confidence"]), json.dumps(md),
                now_iso(), note["id"]))
    db.execute("INSERT INTO memory_edge (id, project, subject_id, predicate, "
               "object_id, source_id) VALUES (?, ?, ?, 'supported_by', ?, ?)",
               (uuid.uuid4().hex[:24], note["project"], note["id"], ev["id"],
                ev["id"]))
    db.commit()
    db.close()
    log_md(args.db, {"op": "promote", "id": args.id, "evidence": args.evidence,
                     "at": md["promoted_at"]},
           f"promote {args.id} on evidence {args.evidence}")
    print(f"promoted {args.id} to fact on evidence {args.evidence}")
    return 0


# ---------------------------------------------------------------- sleep
#
# The distill ("synthetic sleep"), run a slice per turn since v5:
#   1. collect  live raw turns (project != semantic) older than --min-age-hours
#   2. distill  each turn at/above SLEEP_MIN_SALIENCE into ONE semantic entry,
#               linked to the turn and its superseded predecessors by
#               `distilled_from` edges (not `derived_from`, so wiping a turn
#               never cascades into the entry built from it)
#   3. verify   the semantic entry exists and is live, then
#   4. wipe     the consumed turns (reversible tombstone, logged as op=wipe)
#   5. sweep    gc --apply, then dedup --apply
#
# Distillation is 1:1 (one live turn, plus whatever it already superseded,
# becomes one entry). That keeps corrections exact: a later correction aimed
# at a wiped turn is redirected to the entry that carries its fact.
#
# Provenance is computed by the kit, never by the distiller: the entry is a
# fact only if the live turn is user-said or tool-observed, else a note.
#
# Crash safety: every step is idempotent. If a run dies after writing the
# entry but before the wipe, the next run finds the existing entry (by its
# distilled_from edge) and only wipes. If the wipe happened but its log line
# did not, a reindex brings the turn back and the next run wipes it again.

# Tolerant on purpose: case, and whitespace anywhere inside the brackets.
_HEADER = re.compile(r"^\s*\[\s*turn\s+provenance\s*[=:]\s*[^\]]*\]\s*$",
                     re.IGNORECASE)


_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))


def _looks_like_header(line):
    # Match on a normalized copy (NFKC folds fullwidth brackets and the like,
    # zero-width characters are dropped); the stored text itself is untouched.
    return bool(_HEADER.match(unicodedata.normalize("NFKC", line).translate(_ZERO_WIDTH)))


def _body(content):
    """Strip every leading provenance-header line. The kit writes exactly one
    header itself; any header-shaped line in distilled text is forged or
    stale and must not reach search display."""
    lines = (content or "").split("\n")
    while lines and _looks_like_header(lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


def _run_distiller(cmd, payload):
    """Optional external distiller: JSON on stdin, JSON {"text": ...} out.

    Anything it returns is text only. It cannot set provenance, type,
    confidence, or which turns get wiped."""
    import subprocess
    res = subprocess.run(cmd, shell=True, input=json.dumps(payload),
                         capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"distiller exit {res.returncode}: {res.stderr[:200]}")
    out = json.loads(res.stdout)
    text = _body(out.get("text") or "")
    if len(text) < 10:
        raise RuntimeError("distiller returned empty text")
    return text


CONFLICT_SIM = float(os.environ.get("MEMORYKIT_CONFLICT_SIM", "0.85"))


def find_conflicts(db, text, vec):
    """Live semantic entries that say nearly the same thing with a different
    number or flipped negation. Flagged, never auto-resolved: deciding which
    is true needs evidence, not similarity. Costs no tokens."""
    out = []
    for r in db.execute("SELECT id, content, embedding FROM memory_item WHERE "
                        "project=? AND status != 'rejected' AND superseded_by "
                        "IS NULL", (SEMANTIC_PROJECT,)).fetchall():
        other = _body(r["content"])
        if vec and r["embedding"]:
            sim = cosine(vec, json.loads(r["embedding"]))
        else:
            sim = difflib.SequenceMatcher(None, text.lower(), other.lower()).ratio()
        if sim < CONFLICT_SIM:
            continue
        if value_change(text, other):
            out.append(r["id"])
    return out


def insert_semantic(db, db_path, eid, at, text, provenance, salience, turns,
                    supersession_key="", protected=False, source="", quote=None):
    if db.execute("SELECT 1 FROM memory_item WHERE id=?", (eid,)).fetchone():
        return  # idempotent: replay or a re-logged record
    mtype, confidence = PROVENANCE[provenance]
    md = {"provenance": provenance, "captured_at": at, "distilled": True,
          "distilled_from": turns, "salience_reason": "inherited"}
    if supersession_key:
        md["supersession_key"] = supersession_key
    if protected:
        md["protected"] = True
    if source:
        md["source"] = source
    if quote:
        md["quote"] = quote  # metadata only: never indexed as this speaker's words
    vecs = embed_texts([text[:4000]])
    conflicts = find_conflicts(db, text, vecs[0] if vecs else None)
    if conflicts:
        md["possible_conflict_with"] = conflicts
    db.execute(
        """INSERT INTO memory_item
           (id, created, updated, project, session_key, type, content,
            salience, confidence, metadata, embedding, status,
            valid_from, learned_at, last_accessed_at)
           VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, 'confirmed', ?, ?, ?)""",
        (eid, at, at, SEMANTIC_PROJECT, mtype,
         f"[turn provenance={provenance}]\n{text}", salience, confidence,
         json.dumps(md), json.dumps(vecs[0]) if vecs else None, at, at, at))
    for t in turns:
        db.execute(
            "INSERT INTO memory_edge (id, created, updated, project, subject_id,"
            " predicate, object_id, source_id) VALUES (?, ?, ?, ?, ?,"
            " 'distilled_from', ?, ?)",
            (uuid.uuid4().hex[:24], at, at, SEMANTIC_PROJECT, eid, t, eid))


def wipe_turns(db, parent, turns, at):
    """Tombstone consumed turns. Refuses unless the parent is live and every
    turn is linked to it by a distilled_from edge. Never cascades."""
    p = db.execute("SELECT status, superseded_by FROM memory_item WHERE id=?",
                   (parent,)).fetchone()
    if not p or p["status"] == "rejected" or p["superseded_by"]:
        raise RuntimeError(f"refusing wipe: parent {parent} is not live")
    linked = {r["object_id"] for r in db.execute(
        "SELECT object_id FROM memory_edge WHERE subject_id=? AND "
        "predicate='distilled_from'", (parent,)).fetchall()}
    missing = [t for t in turns if t not in linked]
    if missing:
        raise RuntimeError(f"refusing wipe: {missing} not distilled into {parent}")
    wiped = []
    for t in turns:
        r = db.execute("SELECT status, metadata FROM memory_item WHERE id=?",
                       (t,)).fetchone()
        if not r:
            continue
        md = json.loads(r["metadata"] or "{}")
        md["distilled_into"] = parent
        if r["status"] != "rejected":
            md["tombstoned_at"] = at
            md["tombstone_reason"] = f"distilled into {parent}"
            wiped.append(t)
        db.execute("UPDATE memory_item SET status='rejected', metadata=?, "
                   "updated=? WHERE id=?", (json.dumps(md), at, t))
    return wiped


def _predecessors(db, tid):
    out, frontier = [], [tid]
    while frontier:
        cur = frontier.pop()
        for r in db.execute("SELECT id FROM memory_item WHERE superseded_by=?",
                            (cur,)).fetchall():
            if r["id"] not in out:
                out.append(r["id"])
                frontier.append(r["id"])
    return out


def is_settled(db, t):
    """A raw turn is settled when its session closed, or when SETTLE_TURNS
    newer raw captures exist. Function, not clock."""
    ended = set(state_get(db, "ended_sessions", []))
    if t["session_key"] and t["session_key"] in ended:
        return True
    newer = db.execute(
        "SELECT COUNT(*) FROM memory_item WHERE project != ? AND learned_at > ?",
        (SEMANTIC_PROJECT, t["learned_at"])).fetchone()[0]
    return newer >= SETTLE_TURNS


# ---------------------------------------------------------------- v5: per-turn
#
# There is no end-of-day pass. Everything happens inside the two hooks an
# agent already calls every turn:
#
#   pre_turn(query)   start of turn: finish any half-done write left by a
#                     crash, then search and return hits within a token budget.
#   post_turn(turns)  end of turn: journal the turns (fsync), apply them
#                     idempotently, then run ONE bounded upkeep step.
#
# Upkeep is amortized: each post_turn distills at most a couple of settled
# turns, GC-scores a small window of rows from a rolling cursor, and
# dedup-checks a few more. Decay is never "run": gc_score computes it from
# timestamps at the moment a row is looked at. Every step is its own SQLite
# transaction and safe to repeat, so a kill at any point loses nothing and
# the next call (pre or post, any session, any day) picks up where it was.

SETTLE_HOURS = float(os.environ.get("MEMORYKIT_SETTLE_HOURS", "6"))
UPKEEP_DISTILL = int(os.environ.get("MEMORYKIT_UPKEEP_DISTILL", "2"))
UPKEEP_GC_ROWS = int(os.environ.get("MEMORYKIT_UPKEEP_GC_ROWS", "25"))
UPKEEP_DEDUP_ROWS = int(os.environ.get("MEMORYKIT_UPKEEP_DEDUP_ROWS", "8"))
JOURNAL_COMPACT_BYTES = int(os.environ.get("MEMORYKIT_JOURNAL_COMPACT", "65536"))
# Test hook only: kill the process hard at a named point.
_CRASH_AT = os.environ.get("MEMORYKIT_TEST_CRASH_AT", "")


def _crash(point):
    if _CRASH_AT == point:
        os._exit(137)


_LOCKS = {}


@contextlib.contextmanager
def writer_lock(db_path, timeout=60.0):
    """One writer per database across processes (flock on <db>.lock).
    The kernel drops it when the process dies, so a crash can never leave a
    stale lock behind. Re-entrant within a process."""
    key = os.path.abspath(db_path)
    ent = _LOCKS.get(key)
    if ent:
        ent[1] += 1
        try:
            yield
        finally:
            ent[1] -= 1
        return
    try:
        import fcntl
    except ImportError:  # non-POSIX: fall back to SQLite's own busy handling
        yield
        return
    os.makedirs(os.path.dirname(key), exist_ok=True)
    fd = os.open(key + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.time() > deadline:
                os.close(fd)
                raise TimeoutError(f"writer lock busy for {timeout}s: {key}.lock")
            time.sleep(0.05)
    _LOCKS[key] = [fd, 1]
    try:
        yield
    finally:
        _LOCKS.pop(key, None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def turn_id(t):
    """Deterministic id for a turn: retrying the same turn never duplicates."""
    import hashlib
    key = json.dumps([t.get("session", ""), t.get("turn", ""), t.get("speaker", ""),
                      t.get("source", ""), (t.get("text") or "").strip()],
                     ensure_ascii=False)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def _journal_path(db_path):
    return db_path + ".journal.jsonl"


def _journal_append(db_path, objs):
    with open(_journal_path(db_path), "a", encoding="utf-8") as fh:
        for o in objs:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _journal_read(db_path):
    """(pending records in order, set of done ids). A torn last line from a
    crash mid-write is skipped: its turn was never acknowledged."""
    path = _journal_path(db_path)
    recs, done = {}, set()
    if not os.path.exists(path):
        return [], done
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if "done" in o:
                done.add(o["done"])
            elif o.get("jid"):
                recs.setdefault(o["jid"], o)
    return [r for j, r in recs.items() if j not in done], done


def _log_from_row(db_path, eid):
    """Write the markdown-log record for a stored capture, rebuilt from the
    row itself, so a crash between commit and log is repaired exactly."""
    db = connect(db_path)
    r = db.execute("SELECT * FROM memory_item WHERE id=?", (eid,)).fetchone()
    db.close()
    if not r:
        return False
    md = json.loads(r["metadata"] or "{}")
    text = _body(r["content"])
    speaker = {"user-said": "user", "tool-observed": "tool"}.get(
        md.get("provenance"), "agent")
    log_md(db_path, {
        "op": "capture", "id": eid, "at": r["learned_at"], "speaker": speaker,
        "provenance": md.get("provenance", "agent-said"), "text": text,
        "salience": r["salience"], "salience_manual": md.get("salience_reason") == "manual",
        "project": r["project"], "session": r["session_key"] or "",
        "turn": md.get("turn") or "", "supersedes": md.get("supersedes", []),
        "supersession_key": md.get("supersession_key", ""),
        "parents": md.get("parent_ids", []), "protected": bool(md.get("protected")),
        "novelty": md.get("novelty", ""), "source": md.get("source", ""),
        "quote": md.get("quote", {})},
        f"[{md.get('provenance', 'agent-said')}] {text}")
    return True


def _apply_turn(db_path, rec):
    """Store one journaled turn (no log, no upkeep). Idempotent by jid."""
    ns = argparse.Namespace(
        db=db_path, speaker=rec.get("speaker", "agent"), text=rec.get("text", ""),
        session=rec.get("session", ""), turn=rec.get("turn", ""),
        provenance=rec.get("provenance") or None, salience=rec.get("salience"),
        supersedes=rec.get("supersedes", ""), supersession_key=rec.get("supersession_key", ""),
        parent_ids=rec.get("parent_ids", ""), check=False,
        project=rec.get("project", "turns"), protected=bool(rec.get("protected")),
        source=rec.get("source", ""), quote=rec.get("quote", ""),
        quote_source=rec.get("quote_source", ""), no_auto=True, no_log=True,
        force_id=rec["jid"], force_at=rec.get("at"))
    with contextlib.redirect_stdout(io.StringIO()):
        return cmd_capture(ns)


def replay_journal(db_path):
    """Finish every journaled turn that a killed call left unfinished:
    store it if the row is missing, log it, mark it done. Returns how many
    turns it completed. Safe to call any time, any number of times."""
    if not os.path.exists(_journal_path(db_path)):
        return 0
    with writer_lock(db_path):
        return _replay_locked(db_path)


def _replay_locked(db_path):
    pending, _ = _journal_read(db_path)
    n = 0
    for rec in pending:
        jid = rec["jid"]
        db = connect(db_path)
        have = db.execute("SELECT 1 FROM memory_item WHERE id=?", (jid,)).fetchone()
        db.close()
        if not have:
            try:
                rc = _apply_turn(db_path, rec)  # a too-short turn stores nothing
            except sqlite3.IntegrityError:
                rc = 0  # another process stored it first: same id, same turn
            if rc not in (0, None):
                _journal_append(db_path, [{"done": jid, "rc": rc}])
                continue  # refused (e.g. provenance firewall): never retried
        _crash("before_log")
        _log_from_row(db_path, jid)  # no-op when nothing was stored
        _journal_append(db_path, [{"done": jid}])
        n += 1
    _journal_compact(db_path)
    return n


def _journal_compact(db_path):
    """Drop finished records once the journal grows. Atomic rename."""
    path = _journal_path(db_path)
    try:
        if os.path.getsize(path) < JOURNAL_COMPACT_BYTES:
            return
    except OSError:
        return
    pending, _ = _journal_read(db_path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in pending:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _feedback_map(db, ids):
    out = {}
    if not ids:
        return out
    q = ",".join("?" * len(ids))
    for f in db.execute(
            f"SELECT memory_id, rating, COUNT(*) c FROM memory_feedback "
            f"WHERE memory_id IN ({q}) GROUP BY memory_id, rating", ids).fetchall():
        pos, neg = out.get(f["memory_id"], (0, 0))
        if f["rating"] == "+1":
            pos = f["c"]
        elif f["rating"] == "-1":
            neg = f["c"]
        out[f["memory_id"]] = (pos, neg)
    return out


def _window(db, cursor_key, n):
    """Next n live rows after the cursor (by rowid), wrapping at the end."""
    cur = state_get(db, cursor_key, 0)
    rows = db.execute(
        "SELECT rowid AS rid, * FROM memory_item WHERE rowid > ? AND "
        "status != 'rejected' AND superseded_by IS NULL ORDER BY rowid LIMIT ?",
        (cur, n)).fetchall()
    nxt = rows[-1]["rid"] if len(rows) == n else 0  # short read: wrap
    return rows, nxt


def gc_step(db_path, n=None):
    """Score the next n rows; quarantine (reversibly) those under the floor."""
    n = n or UPKEEP_GC_ROWS
    db = connect(db_path)
    rows, nxt = _window(db, "gc_cursor", n)
    fb = _feedback_map(db, [r["id"] for r in rows])
    at, out = now_iso(), []
    for r in rows:
        md = json.loads(r["metadata"] or "{}")
        if is_exempt(r, md):
            continue
        sc = gc_score(r, fb)
        if sc < GC_NOISE_FLOOR:
            tombstone(db, r["id"], f"gc score {sc:.4f} below floor", at=at)
            out.append((r["id"], sc))
    state_set(db, "gc_cursor", nxt)
    db.commit()
    db.close()
    for eid, sc in out:
        log_md(db_path, {"op": "quarantine", "id": eid, "at": at,
                         "reason": f"gc score {sc:.4f} below floor"},
               f"quarantine {eid}")
    return len(out)


def dedup_step(db_path, n=None):
    """Check the next n rows against the whole live set; merge true dups
    through the same gates as `dedup` (cosine, text, identifier guard)."""
    n = n or UPKEEP_DEDUP_ROWS
    db = connect(db_path)
    probe, nxt = _window(db, "dedup_cursor", n)
    merged = []
    if probe:
        live = live_rows(db)
        at = now_iso()
        gone = set()
        for p in probe:
            if p["id"] in gone:
                continue
            pv = load_vec(p)
            if pv:
                sims = _doc_sims(db, live, pv)
                cands = [live[i] for i in sorted(range(len(live)), key=lambda i: -sims[i])[:6]
                         if sims[i] >= DEDUP_COSINE]
            else:
                key = _norm_text(p["content"])
                cands = [r for r in live if not load_vec(r) and _norm_text(r["content"]) == key]
            for c in cands:
                if c["id"] == p["id"] or c["id"] in gone or c["type"] != p["type"]:
                    continue
                if pv:
                    txt = difflib.SequenceMatcher(None, (p["content"] or "")[:4000],
                                                  (c["content"] or "")[:4000]).ratio()
                    if txt < DEDUP_TEXT or identifier_only_diff(p["content"] or "",
                                                                c["content"] or ""):
                        continue
                keep, drop = sorted([p, c], key=provenance_rank)
                db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?",
                           (keep["id"], drop["id"]))
                tombstone(db, drop["id"], f"dedup merged into {keep['id']}", at=at)
                gone.add(drop["id"])
                merged.append((keep["id"], drop["id"]))
                if drop["id"] == p["id"]:
                    break
    state_set(db, "dedup_cursor", nxt)
    db.commit()
    db.close()
    for keep, drop in merged:
        log_md(db_path, {"op": "supersede", "old": drop, "new": keep,
                         "reason": "dedup", "at": now_iso()}, f"dedup {drop} -> {keep}")
    return len(merged)


def upkeep_step(db_path, distill=None):
    """One bounded slice of maintenance. Never raises, never needs a lock
    held across calls, and costs the same whether the last call was a
    minute or a month ago."""
    try:
        with writer_lock(db_path, timeout=5.0):
            return _upkeep_locked(db_path, distill)
    except TimeoutError:
        return {}  # another process is writing; the next turn does this slice


def _upkeep_locked(db_path, distill=None):
    res = {}
    for name, fn in (
            ("distilled", lambda: _distill_step(db_path, distill)),
            ("quarantined", lambda: gc_step(db_path)),
            ("merged", lambda: dedup_step(db_path))):
        try:
            res[name] = fn()
        except Exception as e:  # one failing slice must not stop the turn
            res[name] = 0
            print(f"upkeep {name} skipped: {e}", file=sys.stderr)
    try:
        db = connect(db_path)
        state_set(db, "upkeep_last", {"at": now_iso(), **res})
        db.commit()
        db.close()
    except sqlite3.Error:
        pass
    if VERBOSE and any(res.values()):
        print("upkeep: " + " ".join(f"{k}={v}" for k, v in res.items() if v))
    return res


def _distill_step(db_path, limit=None):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cmd_sleep(argparse.Namespace(
            db=db_path, apply=True, min_age_hours=0.0,
            min_salience=SLEEP_MIN_SALIENCE, distiller=AUTO_DISTILLER,
            no_sweep=True, settled_only=True, quiet_lock=True,
            limit=limit or UPKEEP_DISTILL))
    m = re.search(r"distilled (\d+)", out.getvalue())
    return int(m.group(1)) if m else 0


# ------------------------------------------------- v5: the two-hook contract

def pre_turn(db_path, query, budget=250, prefer_source="", limit=6,
             chars=220, repair=False):
    """Start of turn. Repairs any interrupted write, then returns the hits
    for `query` as terse lines within ~`budget` tokens. With prefer_source,
    up to two slots (and ~40% of the budget) go to that speaker first."""
    if not os.path.exists(db_path):
        init_db_quiet(db_path)
    replay_journal(db_path)
    _maybe_repair(db_path, force=repair)
    if not (query or "").strip():
        return ""
    with writer_lock(db_path):  # search bumps access stats: one writer at a time
        return _pre_turn_search(db_path, query, budget, prefer_source, limit, chars)


REPAIR_EVERY = int(os.environ.get("MEMORYKIT_REPAIR_EVERY", "0"))


def _drift(db_path):
    recs = _read_log_records(md_dir_for(db_path))
    db = connect(db_path)
    db_ids = {r[0] for r in db.execute("SELECT id FROM memory_item")}
    db.close()
    log_ids = _log_capture_ids(recs)
    log_only = {r["id"] for r, _, _ in recs if r.get("op") == "capture"} - db_ids
    return sorted(log_only), sorted(db_ids - log_ids)


def _maybe_repair(db_path, force=False, trigger="pre-turn"):
    """Drift repair inside the turn, only when asked: every
    MEMORYKIT_REPAIR_EVERY turns (0 = never, the default) or with force.
    A full pass reads the whole log, so a large store should not pay for it
    every turn. When it repairs anything it says so on stderr and logs a
    repair_run record naming every id it touched. Never raises."""
    try:
        if not force:
            if REPAIR_EVERY <= 0:
                return None
            db = connect(db_path)
            n = int(state_get(db, "repair_turns", 0) or 0) + 1
            due = n >= REPAIR_EVERY
            state_set(db, "repair_turns", 0 if due else n)
            db.commit()
            db.close()
            if not due:
                return None
        log_only, db_only = _drift(db_path)
        if not log_only and not db_only:
            return {"restored": [], "logged": []}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cmd_repair(argparse.Namespace(db=db_path, md_dir="", apply=True,
                                          drift=True, dedupe_log=False))
        restored = re.findall(r"^\s+restored (\S+)", out.getvalue(), re.M)
        logged = re.findall(r"^\s+logged\s+(\S+)", out.getvalue(), re.M)
        bak = re.search(r"^backup: (.+)$", out.getvalue(), re.M)
        log_md(db_path, {"op": "repair_run", "at": now_iso(), "trigger": trigger,
                         "restored": restored, "logged": logged,
                         "backup": bak.group(1) if bak else ""},
               f"{trigger} repair: restored {len(restored)}, logged {len(logged)}")
        print(f"memorykit: {trigger} repair restored {len(restored)}, logged "
              f"{len(logged)}: " + " ".join(restored + logged), file=sys.stderr)
        return {"restored": restored, "logged": logged}
    except Exception as e:  # a repair problem must not cost the turn its recall
        print(f"memorykit: pre-turn repair skipped: {e}", file=sys.stderr)
        return None


def _pre_turn_search(db_path, query, budget, prefer_source, limit, chars):
    db = connect(db_path)
    lines, ids = [], []
    if prefer_source:
        l1, i1 = search_lines(db, query, 2, chars, budget * 2 // 5, prefer_source)
        lines, ids = l1, i1
    used = sum(len(x) // 4 + 1 for x in lines)
    l2, _ = search_lines(db, query, max(0, limit - len(lines)), chars,
                         max(1, budget - used), "", skip_ids=tuple(ids))
    db.close()
    return "\n".join(lines + l2)


def post_turn(db_path, turns, upkeep=True):
    """End of turn. `turns` is a list of dicts: speaker (user|agent|tool),
    text, and optionally session, turn, source, provenance, salience,
    supersedes, supersession_key, parent_ids, protected, quote, id.
    Journals first (fsync), applies idempotently, then one upkeep step.
    Returns the stored ids (retries of the same turn return the same id)."""
    if not os.path.exists(db_path):
        init_db_quiet(db_path)
    replay_journal(db_path)  # finish anything a previous kill left
    recs, ids = [], []
    for t in turns:
        if (t.get("text") or "").strip() == "":
            continue
        jid = t.get("id") or turn_id(t)
        rec = {k: v for k, v in t.items() if k != "id"}
        rec.update({"jid": jid, "at": t.get("at") or now_iso()})
        recs.append(rec)
        ids.append(jid)
    if recs:
        _journal_append(db_path, recs)
    _crash("after_journal")
    replay_journal(db_path)
    if upkeep and AUTO:
        upkeep_step(db_path)
    return ids


def init_db_quiet(db_path):
    with contextlib.redirect_stdout(io.StringIO()):
        init_db(db_path)


def cmd_pre_turn(args):
    out = pre_turn(args.db, args.query, args.budget, args.prefer_source,
                   args.limit, args.chars, repair=args.repair)
    print(out or ("(no matches)" if args.query.strip() else ""))
    return 0


def cmd_post_turn(args):
    turns = []
    if args.jsonl:
        for line in sys.stdin:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    elif args.text:
        t = {"speaker": args.speaker, "text": args.text}
        for k in ("session", "turn", "source", "provenance", "salience",
                  "supersedes", "supersession_key", "parent_ids", "quote"):
            v = getattr(args, k, None)
            if v not in (None, ""):
                t[k] = v
        if args.protected:
            t["protected"] = True
        turns.append(t)
    for t in turns:
        if t.get("speaker") == "agent" and t.get("provenance") not in (
                None, "", "agent-said", "agent-inferred"):
            print(f"REFUSED: speaker=agent cannot claim provenance={t['provenance']}",
                  file=sys.stderr)
            return 2
    ids = post_turn(args.db, turns, upkeep=not args.no_upkeep)
    print(f"stored {len(ids)}" + (f" last={short(ids[-1])}" if ids else ""))
    return 0


def cmd_upkeep(args):
    replay_journal(args.db)
    if not getattr(args, "no_repair", False):
        r = _maybe_repair(args.db, force=True, trigger="upkeep")
        if r and (r["restored"] or r["logged"]):
            print(f"repair: restored {len(r['restored'])}, logged {len(r['logged'])} "
                  "(named in a repair_run log record)")
    tot = {}
    for _ in range(max(1, args.steps)):
        for k, v in upkeep_step(args.db).items():
            tot[k] = tot.get(k, 0) + v
    print(json.dumps(tot))
    return 0


def cmd_upgrade(args):
    """Migrate an existing install in place: back up, add what v5 needs,
    switch to WAL, finish any journal. Idempotent; data never moves."""
    path = args.db
    if not os.path.exists(path):
        init_db_quiet(path)
        print(f"upgrade: no database at {path}; created a fresh v5 one")
        return 0
    src = sqlite3.connect(path)
    try:
        before = state_get(src, "schema_version", None)
    except sqlite3.Error:
        before = None
    if not args.no_backup and before != 5:
        bak = f"{path}.pre-v5-{datetime.datetime.now():%Y%m%d-%H%M%S}.bak"
        dst = sqlite3.connect(bak)
        src.backup(dst)
        dst.close()
        print(f"backup: {bak}")
    src.close()
    db = connect(path)
    db.executescript(SCHEMA)
    migrate(db)
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    state_set(db, "schema_version", 5)
    if before != 5:
        state_set(db, "upgraded", {"from": before or "4.x", "at": now_iso()})
    n = db.execute("SELECT COUNT(*) FROM memory_item").fetchone()[0]
    db.commit()
    ok = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    lock = path + ".sleep.lock"
    if lock_status(lock)[0] == "stale":
        os.remove(lock)
        print("removed stale sleep lock")
    done = replay_journal(path)
    print(f"upgrade: {path} rows={n} journal_mode={mode} integrity={ok} "
          f"replayed={done} schema=5")
    return 0 if ok == "ok" else 1


def cmd_bench(args):
    """Read-only retrieval check on your own corpus. --queries is a JSON
    list of [query, needle] pairs; a hit is any of the top k whose text
    contains the needle. Tune thresholds on your data, not on ours."""
    pairs = json.load(open(args.queries, encoding="utf-8"))
    db = connect(args.db)
    ranks = []
    for q, needle in pairs:
        hits = search_rows(db, q, args.k)
        ix = [i for i, (r, _, _) in enumerate(hits) if needle in (r["content"] or "")]
        ranks.append(ix[0] + 1 if ix else None)
        if ix == [] and args.show_misses:
            print(f"miss: {q[:80]}")
    db.close()
    n = len(ranks) or 1
    at = lambda k: sum(1 for r in ranks if r and r <= k) / n
    mrr = sum(1 / r for r in ranks if r) / n
    print(json.dumps({"queries": len(ranks), "R@1": round(at(1), 3),
                      "R@3": round(at(3), 3), "R@5": round(at(5), 3),
                      "MRR": round(mrr, 3)}))
    return 0


def cmd_health(args):
    """Read-only. One JSON object; exit 0 healthy, 1 if anything needs a look."""
    db = connect(args.db)
    q = lambda sql, *a: db.execute(sql, a).fetchone()[0]
    raw_live = q("SELECT COUNT(*) FROM memory_item WHERE project != ? AND "
                 "status != 'rejected' AND superseded_by IS NULL", SEMANTIC_PROJECT)
    sel = argparse.Namespace(db=args.db, apply=False, min_age_hours=0.0,
                             min_salience=SLEEP_MIN_SALIENCE, distiller="",
                             no_sweep=True, settled_only=True)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _sleep(sel)
    backlog = sum(1 for ln in out.getvalue().splitlines()
                  if ln.startswith("would distill") and "[reuse]" not in ln)
    reuse = q("SELECT COUNT(DISTINCT t.id) FROM memory_item t JOIN memory_edge e "
              "ON e.object_id = t.id AND e.predicate='distilled_from' "
              "JOIN memory_item p ON p.id = e.subject_id "
              "WHERE t.status != 'rejected' AND p.status != 'rejected' "
              "AND p.superseded_by IS NULL")
    oldest = db.execute(
        "SELECT MIN(learned_at) FROM memory_item WHERE project != ? AND "
        "status != 'rejected' AND superseded_by IS NULL",
        (SEMANTIC_PROJECT,)).fetchone()[0]
    db_ids = {r[0] for r in db.execute("SELECT id FROM memory_item")}
    rows = len(db_ids)
    _recs, bad_lines = _scan_log(md_dir_for(args.db), quiet=True)
    seen = _log_capture_ids(_recs)
    state, info = lock_status(args.db + ".sleep.lock")
    report = {
        "raw_live": raw_live,
        "settled_backlog": backlog,
        "entry_written_not_wiped": reuse,
        "oldest_live_raw": oldest,
        "semantic_live": q("SELECT COUNT(*) FROM memory_item WHERE project=? AND "
                           "status != 'rejected' AND superseded_by IS NULL",
                           SEMANTIC_PROJECT),
        "wiped": q("SELECT COUNT(*) FROM memory_item WHERE "
                   "json_extract(metadata,'$.distilled_into') IS NOT NULL"),
        "quarantined": q("SELECT COUNT(*) FROM memory_item WHERE "
                         "status='rejected' AND json_extract(metadata,'$.tombstone_reason') "
                         "LIKE 'gc score%'"),
        "upkeep_last": state_get(db, "upkeep_last", {}),
        "journal_pending": len(_journal_read(args.db)[0]),
        "possible_conflicts": q("SELECT COUNT(*) FROM memory_item WHERE project=? "
                                "AND status != 'rejected' AND superseded_by IS NULL "
                                "AND json_extract(metadata,'$.possible_conflict_with') "
                                "IS NOT NULL", SEMANTIC_PROJECT),
        "db_rows": rows,
        "log_entries": len(seen),
        "in_db_not_in_log": len(db_ids - seen),
        "in_log_not_in_db": len(seen - db_ids),
        "in_db_not_in_log_ids": sorted(db_ids - seen)[:100],
        "in_log_not_in_db_ids": sorted(seen - db_ids)[:100],
        "unparseable_log_lines": bad_lines[:100],
        "lock": state,
        "lock_info": info,
        "auto": AUTO,
    }
    db.close()
    problems = []
    if state == "stale":
        problems.append("stale lock (next sleep run clears it)")
    if state == "held" and info.get("age_s", 0) > LOCK_TIMEOUT / 2:
        problems.append("lock held a long time")
    if backlog > SETTLE_TURNS * 4:
        problems.append("settled turns piling up: is post-turn being called?")
    if report["journal_pending"]:
        problems.append("journaled turns not applied: the next pre/post-turn finishes them")
    if reuse:
        problems.append("entries written but turns not wiped: run sleep --apply")
    if db_ids - seen:
        problems.append("rows missing from the log are lost on reindex: run repair --apply")
    if seen - db_ids:
        problems.append("log has entries the DB lacks: run repair --apply")
    if bad_lines:
        problems.append("unparseable log lines (see unparseable_log_lines)")
    report["problems"] = problems
    if not getattr(args, "pretty", False):
        report = {k: v for k, v in report.items() if v not in (0, {}, [], None)
                  or k in ("problems", "lock")}
    print(json.dumps(report, ensure_ascii=False,
                     indent=2 if getattr(args, "pretty", False) else None))
    return 1 if problems else 0


def cmd_end_session(args):
    """Optional since v5: nothing depends on it. Closing a session only lets
    its turns settle before the age/turn-count rules would."""
    db = connect(args.db)
    ended = state_get(db, "ended_sessions", [])
    if args.session not in ended:
        ended.append(args.session)
    state_set(db, "ended_sessions", ended)
    db.commit()
    db.close()
    log_md(args.db, {"op": "end_session", "session": args.session},
           f"end session {args.session}")
    print(f"session {args.session} closed")
    if not _REPLAYING:
        return cmd_sleep(argparse.Namespace(
            db=args.db, apply=True, min_age_hours=0.0,
            min_salience=SLEEP_MIN_SALIENCE, distiller=AUTO_DISTILLER,
            no_sweep=True, settled_only=True))
    return 0


LOCK_TIMEOUT = int(os.environ.get("MEMORYKIT_LOCK_TIMEOUT", "3600"))


def lock_status(lock):
    """('none'|'held'|'stale', info). Stale means: same host and the PID is
    gone, or the lock is older than LOCK_TIMEOUT. Read-only."""
    import socket
    if not os.path.exists(lock):
        return "none", {}
    try:
        info = json.loads(open(lock).read())
    except (ValueError, OSError):
        info = {}
    age = time.time() - os.path.getmtime(lock)
    info["age_s"] = int(age)
    pid, host = info.get("pid"), info.get("host")
    if pid and host == socket.gethostname():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return "stale", info
        except PermissionError:
            pass  # alive, owned by someone else
        else:
            if age <= LOCK_TIMEOUT:
                return "held", info
    return ("stale" if age > LOCK_TIMEOUT else "held"), info


def acquire_lock(lock):
    import socket
    for attempt in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(),
                                     "host": socket.gethostname(),
                                     "at": now_iso()}).encode())
            os.close(fd)
            return True, {}
        except FileExistsError:
            state, info = lock_status(lock)
            if state == "stale" and attempt == 1:
                # Clear it, say so, and retry once. Every clear is visible.
                try:
                    os.remove(lock)
                except FileNotFoundError:
                    pass
                print(f"cleared stale sleep lock (pid {info.get('pid')}, "
                      f"age {info.get('age_s')}s)", file=sys.stderr)
                continue
            return False, info
    return False, {}


def cmd_sleep(args):
    lock = args.db + ".sleep.lock"
    if args.apply:
        ok, info = acquire_lock(lock)
        if not ok:
            # Never silent, even from auto-sleep: a held lock means work waits.
            print(f"sleep skipped: lock held by pid {info.get('pid')} for "
                  f"{info.get('age_s')}s ({lock})", file=sys.stderr)
            return 3
    try:
        return _sleep(args)
    finally:
        if args.apply and os.path.exists(lock):
            os.remove(lock)


def _sleep(args):
    db = connect(args.db)
    nowdt = datetime.datetime.now(datetime.timezone.utc)
    cutoff = (nowdt - datetime.timedelta(hours=args.min_age_hours)).isoformat()
    turns = db.execute(
        "SELECT * FROM memory_item WHERE status != 'rejected' AND "
        "superseded_by IS NULL AND project != ? AND learned_at <= ? "
        "ORDER BY learned_at", (SEMANTIC_PROJECT, cutoff)).fetchall()
    if getattr(args, "settled_only", False):
        # One pass instead of a COUNT per turn: a turn is in the working
        # window if it is among the newest SETTLE_TURNS raw captures.
        stamps = [r[0] for r in db.execute(
            "SELECT learned_at FROM memory_item WHERE project != ? "
            "ORDER BY learned_at DESC LIMIT ?",
            (SEMANTIC_PROJECT, SETTLE_TURNS)).fetchall()]
        floor = stamps[-1] if len(stamps) >= SETTLE_TURNS else None
        ended = set(state_get(db, "ended_sessions", []))
        # v5: a turn also settles by age, so a session that never gets an
        # end-session call (crash, offline, forgotten) still drains.
        aged = (nowdt - datetime.timedelta(hours=SETTLE_HOURS)).isoformat()
        turns = [t for t in turns
                 if (t["session_key"] and t["session_key"] in ended)
                 or (floor is not None and t["learned_at"] < floor)
                 or t["learned_at"] < aged]
    distilled = wiped = skipped = reused = errors = 0
    limit = getattr(args, "limit", 0) or 0
    for t in turns:
        if limit and distilled + reused >= limit:
            break  # bounded: the rest waits for the next turn's upkeep
        md = json.loads(t["metadata"] or "{}")
        if (t["salience"] < args.min_salience and not md.get("protected")
                and md.get("hits", 0) < SLEEP_MIN_HITS):
            skipped += 1
            continue
        group = [t["id"]] + _predecessors(db, t["id"])
        existing = db.execute(
            "SELECT e.subject_id FROM memory_edge e JOIN memory_item m "
            "ON m.id = e.subject_id WHERE e.object_id=? AND "
            "e.predicate='distilled_from' AND m.status != 'rejected' AND "
            "m.superseded_by IS NULL", (t["id"],)).fetchone()
        prov = md.get("provenance", "agent-said")
        if prov not in PROVENANCE:
            prov = "agent-said"
        if not args.apply:
            print(f"would distill {t['id']} ({prov}, sal={t['salience']}, "
                  f"{len(group)} turn(s)){' [reuse]' if existing else ''}")
            distilled += 0 if existing else 1
            continue
        try:
            at = now_iso()
            if existing:
                parent = existing["subject_id"]
                reused += 1
                # The entry may have been committed by a run that died before
                # logging it. Re-log it so the markdown can rebuild it (replay
                # of a duplicate record is a no-op).
                er = db.execute("SELECT * FROM memory_item WHERE id=?",
                                (parent,)).fetchone()
                emd = json.loads(er["metadata"] or "{}")
                log_md(args.db, {"op": "distill", "id": parent,
                                 "at": er["learned_at"], "text": _body(er["content"]),
                                 "provenance": emd.get("provenance", prov),
                                 "salience": er["salience"],
                                 "turns": emd.get("distilled_from", group),
                                 "supersession_key": emd.get("supersession_key", ""),
                                 "protected": bool(emd.get("protected")),
                                 "source": emd.get("source", ""),
                                 "quote": emd.get("quote", {}),
                                 "relogged": True},
                       f"[distilled, re-logged] {_body(er['content'])}")
            else:
                text = _body(t["content"])
                if args.distiller:
                    text = _run_distiller(args.distiller, {
                        "turns": [{"id": g, "text": _body(db.execute(
                            "SELECT content FROM memory_item WHERE id=?",
                            (g,)).fetchone()["content"])} for g in group],
                        "provenance": prov})
                parent = uuid.uuid4().hex[:24]
                insert_semantic(db, args.db, parent, at, text, prov,
                                t["salience"], group,
                                md.get("supersession_key", ""),
                                bool(md.get("protected")),
                                md.get("source", ""), md.get("quote") or None)
                db.commit()
                log_md(args.db, {"op": "distill", "id": parent, "at": at,
                                 "text": text, "provenance": prov,
                                 "salience": t["salience"], "turns": group,
                                 "supersession_key": md.get("supersession_key", ""),
                                 "protected": bool(md.get("protected")),
                                 "source": md.get("source", ""),
                                 "quote": md.get("quote", {})},
                       f"[distilled {prov}] {text}")
                distilled += 1
            # verify, then wipe
            at2 = now_iso()
            done = wipe_turns(db, parent, group, at2)
            db.commit()
            log_md(args.db, {"op": "wipe", "parent": parent, "turns": group,
                             "at": at2}, f"wipe {len(done)} turn(s) into {parent}")
            wiped += len(done)
        except Exception as e:
            db.rollback()
            errors += 1
            print(f"ERROR on {t['id']}: {e} (turn left live; rerun is safe)",
                  file=sys.stderr)
    db.close()
    verb = "distilled" if args.apply else "would distill"
    print(f"sleep: {verb} {distilled}, reused {reused}, wiped {wiped}, "
          f"left for gc {skipped}, errors {errors}")
    if args.apply and not args.no_sweep:
        cmd_gc(argparse.Namespace(db=args.db, apply=True, restore="",
                                  eviction_report=False))
        cmd_dedup(argparse.Namespace(db=args.db, apply=True))
    return 1 if errors else 0


def cmd_reindex(args):
    """Rebuild the database from the markdown log alone.

    Replays every mk record in timestamp order through the same code paths
    that wrote it, with logging off. Vectors are recomputed if an embedder
    is installed (otherwise run `backfill` later)."""
    global _REPLAYING
    d = args.md_dir or md_dir_for(args.db)
    files = sorted(f for f in os.listdir(d) if f.endswith(".md")) \
        if os.path.isdir(d) else []
    records, _bad = _scan_log(d)
    if os.path.exists(args.db):
        if not args.force:
            print(f"{args.db} exists; pass --force to replace it", file=sys.stderr)
            return 1
        try:
            c = sqlite3.connect(args.db)
            db_ids = {r[0] for r in c.execute("SELECT id FROM memory_item")}
            c.close()
        except sqlite3.Error:
            db_ids = set()
        lost = sorted(db_ids - _log_capture_ids(records))
        if lost and not args.drop_missing:
            print(f"refusing: {len(lost)} row(s) are in the DB but not in the log "
                  f"and would be lost:", file=sys.stderr)
            for i in lost:
                print(f"  would drop {i}", file=sys.stderr)
            print("run `repair --apply` first to write them back to the log, or pass "
                  "--drop-missing to rebuild without them", file=sys.stderr)
            return 2
        for i in lost:
            print(f"dropping {i} (not in log)", file=sys.stderr)
        os.remove(args.db)
    init_db(args.db)
    _REPLAYING = True
    counts = {}
    distilled = {r["id"] for r, _, _ in records if r.get("op") == "distill" and r.get("id")}
    built, dup = set(), []
    try:
        for rec, f, n in records:
            op = rec.get("op")
            if op in ("capture", "distill"):
                # 5.0.2 repair could log a distilled row a second time as a
                # capture; the distill record is the real one.
                if rec.get("id") in built or (op == "capture" and rec.get("id") in distilled):
                    dup.append((rec.get("id"), op, f"{f}:{n + 1}"))
                    continue
                built.add(rec.get("id"))
            counts[op] = counts.get(op, 0) + 1
            if op == "capture":
                ns = argparse.Namespace(
                    db=args.db, speaker=rec["speaker"], text=rec["text"],
                    session=rec.get("session", ""), turn=rec.get("turn", ""),
                    provenance=rec["provenance"],
                    salience=rec["salience"], supersedes=",".join(
                        rec.get("supersedes", [])),
                    supersession_key="", parent_ids=",".join(
                        rec.get("parents", [])), check=False,
                    project=rec.get("project", "turns"),
                    protected=rec.get("protected", False),
                    source=rec.get("source", ""),
                    quote=(rec.get("quote") or {}).get("text", ""),
                    quote_source=(rec.get("quote") or {}).get("source", ""),
                    force_id=rec["id"], force_at=rec["at"])
                with open(os.devnull, "w") as devnull:
                    old = sys.stdout
                    sys.stdout = devnull
                    try:
                        cmd_capture(ns)
                    finally:
                        sys.stdout = old
                db = connect(args.db)
                md = json.loads(db.execute(
                    "SELECT metadata FROM memory_item WHERE id=?",
                    (rec["id"],)).fetchone()[0])
                if rec.get("supersession_key"):
                    md["supersession_key"] = rec["supersession_key"]
                md["salience_reason"] = "manual" if rec.get(
                    "salience_manual") else "heuristic"
                md["novelty"] = rec.get("novelty", md.get("novelty"))
                md["reindexed"] = True
                db.execute("UPDATE memory_item SET metadata=?, last_accessed_at=?"
                           " WHERE id=?", (json.dumps(md), rec["at"], rec["id"]))
                db.commit()
                db.close()
            elif op == "supersede":
                db = connect(args.db)
                db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?",
                           (rec["new"], rec["old"]))
                tombstone(db, rec["old"], f"superseded by {rec['new']}",
                          at=rec.get("at"))
                db.commit()
                db.close()
            elif op == "quarantine":
                db = connect(args.db)
                tombstone(db, rec["id"], rec.get("reason", "quarantine"),
                          at=rec.get("at"))
                db.commit()
                db.close()
            elif op == "relate":
                db = connect(args.db)
                try:
                    relate(db, rec["a"], rec["b"], rec["predicate"], rec["weight"], rec["at"])
                    db.commit()
                except ValueError as e:
                    print(f"replay: {e}", file=sys.stderr)
                db.close()
            elif op == "weigh":
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    cmd_weigh(argparse.Namespace(db=args.db, id=rec["id"],
                                                 salience=rec["salience"], delta=None,
                                                 force_at=rec["at"]))
            elif op == "merge":
                db = connect(args.db)
                merge(db, rec["keep"], rec["drop"], rec["at"])
                db.commit()
                db.close()
            elif op == "restore":
                db = connect(args.db)
                restore(db, rec["id"], rec.get("at"))
                db.commit()
                db.close()
            elif op == "feedback":
                cmd_feedback(argparse.Namespace(
                    db=args.db, id=rec["id"], rating=rec["rating"],
                    comment=rec.get("comment") or None, force_at=rec["at"]))
            elif op == "promote":
                cmd_promote(argparse.Namespace(
                    db=args.db, id=rec["id"], evidence=rec["evidence"],
                    force_at=rec["at"]))
            elif op == "distill":
                db = connect(args.db)
                insert_semantic(db, args.db, rec["id"], rec["at"], rec["text"],
                                rec["provenance"], rec["salience"], rec["turns"],
                                rec.get("supersession_key", ""),
                                rec.get("protected", False),
                                rec.get("source", ""), rec.get("quote") or None)
                db.commit()
                db.close()
            elif op == "end_session":
                db = connect(args.db)
                ended = state_get(db, "ended_sessions", [])
                if rec["session"] not in ended:
                    ended.append(rec["session"])
                state_set(db, "ended_sessions", ended)
                db.commit()
                db.close()
            elif op == "wipe":
                db = connect(args.db)
                try:
                    wipe_turns(db, rec["parent"], rec["turns"], rec["at"])
                    db.commit()
                except RuntimeError as e:
                    print(f"replay: {e}", file=sys.stderr)
                db.close()
    finally:
        _REPLAYING = False
    print(f"reindexed {len(records)} record(s) from {len(files)} file(s): "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if dup:
        print(f"skipped {len(dup)} duplicate record(s) for rows already built "
              "(e.g. distilled rows re-logged by 5.0.2 repair); "
              "`repair --dedupe-log` removes the ones it can prove are copies:")
        for eid, op, loc in dup:
            print(f"  skipped duplicate {op} {eid} at {loc}")
    return 0


def _scan_log(d, quiet=False):
    """Every mk record in the markdown log, sorted by time, plus the lines that
    look like records but cannot be parsed (torn writes, hand edits). Each bad
    line is reported on stderr as <file>:<line> unless quiet."""
    recs, bad = [], []
    files = sorted(f for f in os.listdir(d) if f.endswith(".md")) if os.path.isdir(d) else []
    for f in files:
        for n, line in enumerate(open(os.path.join(d, f), encoding="utf-8", errors="replace")):
            if "<!-- mk:" not in line:
                continue
            m = MK_RE.search(line)
            try:
                if not m:
                    raise ValueError("torn line")
                recs.append((json.loads(m.group(1)), f, n))
            except (ValueError, json.JSONDecodeError):
                bad.append(f"{f}:{n + 1}")
    if bad and not quiet:
        print(f"skipped {len(bad)} unparseable record(s):", file=sys.stderr)
        for loc in bad:
            print(f"  skipped unparseable record at {loc}", file=sys.stderr)
    recs.sort(key=lambda x: (x[0].get("at", ""), x[1], x[2]))
    return recs, bad


def _read_log_records(d):
    return _scan_log(d)[0]


def _log_capture_ids(recs):
    return {r["id"] for r, _, _ in recs
            if r.get("op") in ("capture", "distill") and r.get("id")}


def _norm(t):
    return " ".join((t or "").split())


def _visible_matches(cache, d, f, n, text):
    """The human-readable part of a log line (before <!-- mk: -->), minus the
    bullet and the [tag], must still say what the record's text says. It is
    cut at 300 characters when written, so compare up to its length. An edit
    to the visible line alone means a person touched it: not a provable copy."""
    if f not in cache:
        cache[f] = open(os.path.join(d, f), encoding="utf-8", errors="replace").readlines()
    line = cache[f][n] if n < len(cache[f]) else ""
    vis = line.split("<!-- mk:", 1)[0].strip()
    vis = re.sub(r"^-\s*", "", vis)
    vis = re.sub(r"^\[[^\]]*\]\s*", "", vis)
    vis, want = _norm(vis), _norm(text)
    return bool(vis) and want.startswith(vis) and (len(vis) >= len(want) or len(vis) >= 250)


def _dedupe_plan(db_path, d):
    """Which log records are provable copies of a row, and why the rest stay.

    A record is removed only when all of these hold: another record for the
    same id comes first in the log (the distill record, else the earliest
    capture); the row with that id still exists in the DB; and the copy's
    text, the original's text and the row's text are identical (whitespace
    aside). Anything short of that is kept and named with its reason. No
    label ("db-only", "duplicate") is trusted on its own."""
    recs, _bad = _scan_log(d, quiet=True)
    first = {}
    for rec, f, n in recs:  # distill record is the original when one exists
        if rec.get("op") == "distill" and rec.get("id"):
            first.setdefault(rec["id"], (rec, f, n))
    for rec, f, n in recs:
        if rec.get("op") == "capture" and rec.get("id"):
            first.setdefault(rec["id"], (rec, f, n))
    db = connect(db_path)
    remove, keep = [], []
    lines = {}
    for rec, f, n in recs:
        eid = rec.get("id")
        if rec.get("op") not in ("capture", "distill") or not eid:
            continue
        orig = first.get(eid)
        if orig is None or (orig[1], orig[2]) == (f, n):
            continue  # this record is the original
        loc = f"{f}:{n + 1}"
        row = db.execute("SELECT content FROM memory_item WHERE id=?", (eid,)).fetchone()
        if row is None:
            keep.append((eid, loc, "row not in DB; the copy may be the only trace"))
        elif _norm(rec.get("text")) != _norm(orig[0].get("text")):
            keep.append((eid, loc, "text differs from the original record"))
        elif _norm(rec.get("text")) != _norm(_body(row["content"])):
            keep.append((eid, loc, "text differs from the DB row"))
        elif not _visible_matches(lines, d, f, n, rec.get("text")):
            keep.append((eid, loc, "visible line differs from the record (hand edit?)"))
        else:
            remove.append((eid, f, n, f"{orig[1]}:{orig[2] + 1}"))
    db.close()
    return remove, keep


def cmd_dedupe_log(args):
    d = args.md_dir or md_dir_for(args.db)
    remove, keep = _dedupe_plan(args.db, d)
    print(f"dedupe: {len(remove)} provable duplicate record(s), {len(keep)} kept")
    for eid, f, n, orig in remove:
        print(f"  remove {eid} at {f}:{n + 1} (copy of {orig}; row exists; text matches)")
    for eid, loc, why in keep:
        print(f"  keep   {eid} at {loc}: {why}")
    if not args.apply:
        print("dry run: pass --apply to remove them (the log is backed up first)")
        return 0
    if not remove:
        print("nothing to remove")
        return 0
    with writer_lock(args.db):
        remove, _ = _dedupe_plan(args.db, d)  # re-check under the lock
        bak = f"{d.rstrip(os.sep)}.pre-dedupe-{int(time.time())}"
        shutil.copytree(d, bak)
        print(f"backup: {bak}")
        by_file = {}
        for eid, f, n, orig in remove:
            by_file.setdefault(f, {})[n] = eid
        done = []
        for f, lines in by_file.items():
            path = os.path.join(d, f)
            src = open(path, encoding="utf-8", errors="replace").readlines()
            out = []
            for n, line in enumerate(src):
                eid = lines.get(n)
                m = MK_RE.search(line) if eid else None
                if m and json.loads(m.group(1)).get("id") == eid:
                    done.append((eid, f"{f}:{n + 1}"))
                    continue
                if eid:
                    print(f"  kept {eid} at {f}:{n + 1}: line changed since the plan",
                          file=sys.stderr)
                out.append(line)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(out)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
    for eid, loc in done:
        print(f"  removed {eid} at {loc}")
    log_md(args.db, {"op": "dedupe_log", "at": now_iso(), "backup": bak,
                     "removed": [{"id": e, "at_line": l} for e, l in done]},
           f"dedupe-log removed {len(done)} duplicate record(s); backup {bak}")
    print(f"deduped: removed {len(done)} record(s); the list is logged as a dedupe_log record")
    return 0


def cmd_repair(args):
    """Heal drift between the DB and the markdown log without losing either side.

    Dry run unless --apply. Order: back up the DB, write DB-only rows into the
    log, replay log-only captures into the DB (plus the supersede, quarantine,
    relate and weigh records that touch them). Nothing is dropped; running it
    twice changes nothing the second time."""
    global _REPLAYING
    if getattr(args, "dedupe_log", False):
        return cmd_dedupe_log(args)
    d = args.md_dir or md_dir_for(args.db)
    recs = _read_log_records(d)
    db = connect(args.db)
    db_ids = {r[0] for r in db.execute("SELECT id FROM memory_item")}
    db.close()
    log_ids = _log_capture_ids(recs)  # capture AND distill records both rebuild a row
    missing = [r for r, _, _ in recs if r.get("op") == "capture" and r["id"] not in db_ids]
    orphans = sorted(db_ids - log_ids)
    print(f"drift: {len(missing)} log-only capture(s), {len(orphans)} db-only row(s)")
    for r in missing:
        print(f"  log-only {r['id']}")
    for i in orphans:
        print(f"  db-only  {i}")
    if not args.apply:
        print("dry run: pass --apply to repair (a backup is made first)")
        return 0
    if not missing and not orphans:
        print("nothing to repair")
        return 0
    with writer_lock(args.db):
        bak = f"{args.db}.pre-repair-{int(time.time())}.bak"
        src = sqlite3.connect(args.db)
        dst = sqlite3.connect(bak)
        src.backup(dst)
        dst.close()
        src.close()
        print(f"backup: {bak}")
        for eid in orphans:  # export first: reindex-style rebuilds would lose these
            _log_from_row(args.db, eid)
        new_ids = set()
        _REPLAYING = True
        try:
            for rec in missing:
                ns = argparse.Namespace(
                    db=args.db, speaker=rec["speaker"], text=rec["text"],
                    session=rec.get("session", ""), turn=rec.get("turn", ""),
                    provenance=rec["provenance"], salience=rec["salience"],
                    supersedes=",".join(rec.get("supersedes", [])),
                    supersession_key="", parent_ids=",".join(rec.get("parents", [])),
                    check=False, project=rec.get("project", "turns"),
                    protected=rec.get("protected", False), source=rec.get("source", ""),
                    quote=(rec.get("quote") or {}).get("text", ""),
                    quote_source=(rec.get("quote") or {}).get("source", ""),
                    no_auto=True, no_log=True, force_id=rec["id"], force_at=rec["at"])
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        cmd_capture(ns)
                    except sqlite3.IntegrityError:
                        continue
                db = connect(args.db)
                row = db.execute("SELECT metadata FROM memory_item WHERE id=?", (rec["id"],)).fetchone()
                if row:
                    md = json.loads(row[0] or "{}")
                    if rec.get("supersession_key"):
                        md["supersession_key"] = rec["supersession_key"]
                    md["salience_reason"] = "manual" if rec.get("salience_manual") else "heuristic"
                    md["repaired"] = True
                    db.execute("UPDATE memory_item SET metadata=? WHERE id=?", (json.dumps(md), rec["id"]))
                    db.commit()
                    new_ids.add(rec["id"])
                db.close()
            for rec, _, _ in recs:
                op = rec.get("op")
                db = connect(args.db)
                try:
                    if op == "supersede" and (rec["old"] in new_ids or rec["new"] in new_ids):
                        db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?", (rec["new"], rec["old"]))
                        tombstone(db, rec["old"], f"superseded by {rec['new']}", at=rec.get("at"))
                    elif op == "quarantine" and rec["id"] in new_ids:
                        tombstone(db, rec["id"], rec.get("reason", "quarantine"), at=rec.get("at"))
                    elif op == "relate" and (rec["a"] in new_ids or rec["b"] in new_ids):
                        relate(db, rec["a"], rec["b"], rec["predicate"], rec["weight"], rec["at"])
                    db.commit()
                except (ValueError, KeyError, sqlite3.Error) as e:
                    print(f"repair: skipped {op}: {e}", file=sys.stderr)
                db.close()
        finally:
            _REPLAYING = False
    for i in sorted(new_ids):
        print(f"  restored {i}")
    for i in orphans:
        print(f"  logged   {i}")
    print(f"repaired: restored {len(new_ids)} node(s), logged {len(orphans)} orphan(s)")
    return 0


# ---------------------------------------------------------------- graph ops
#
# Explicit node/edge operations. Nodes are entries; a node's weight is its
# salience (1-10). Edges live in memory_edge with a weight. The system edges
# (derived_from, distilled_from) are written only by the engine; `relate`
# writes every other predicate. Every op is logged to markdown and replayed
# by `reindex`, like any other state change.

OVERLAP_COS = float(os.environ.get("MEMORYKIT_OVERLAP_COS", "0.92"))


def _live_ids(db, ids):
    out = []
    for i in ids:
        r = db.execute("SELECT id, status FROM memory_item WHERE id=?", (i,)).fetchone()
        if not r:
            raise ValueError(f"no entry {i}")
        out.append(r["id"])
    return out


def relate(db, a, b, predicate, weight, at):
    """Upsert one edge a -predicate-> b. weight <= 0 removes it."""
    if predicate in SYSTEM_PREDICATES:
        raise ValueError(f"{predicate} is a system edge; use another predicate")
    if a == b:
        raise ValueError("an entry cannot relate to itself")
    row = db.execute("SELECT id FROM memory_edge WHERE subject_id=? AND object_id=?"
                     " AND predicate=?", (a, b, predicate)).fetchone()
    if weight <= 0:
        if row:
            db.execute("DELETE FROM memory_edge WHERE id=?", (row["id"],))
        return "removed" if row else "absent"
    if row:
        db.execute("UPDATE memory_edge SET weight=?, updated=? WHERE id=?",
                   (weight, at, row["id"]))
        return "updated"
    db.execute("INSERT INTO memory_edge (id, created, updated, project, subject_id,"
               " predicate, object_id, source_id, weight) VALUES (?,?,?,?,?,?,?,?,?)",
               (uuid.uuid4().hex[:24], at, at, "graph", a, predicate, b, None, weight))
    return "added"


def cmd_relate(args):
    db = connect(args.db)
    try:
        a, b = _live_ids(db, [resolve_id(db, args.a), resolve_id(db, args.b)])
        at = getattr(args, "force_at", None) or now_iso()
        res = relate(db, a, b, args.predicate, args.weight, at)
        db.commit()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        db.close()
    log_md(args.db, {"op": "relate", "a": a, "b": b, "predicate": args.predicate,
                     "weight": args.weight, "at": at},
           f"relate {short(a)} -{args.predicate}({args.weight:g})-> {short(b)}")
    print(f"{res} {short(a)} -{args.predicate}({args.weight:g})-> {short(b)}")
    return 0


def cmd_weigh(args):
    """Set a node's weight (salience 1-10). A human/agent judgment, logged."""
    db = connect(args.db)
    r = db.execute("SELECT salience, metadata FROM memory_item WHERE id=?",
                   (args.id,)).fetchone()
    if not r:
        print(f"ERROR: no entry {args.id}", file=sys.stderr)
        return 2
    sal = args.salience if args.salience is not None else r["salience"] + (args.delta or 0)
    sal = max(1, min(10, int(sal)))
    at = getattr(args, "force_at", None) or now_iso()
    md = json.loads(r["metadata"] or "{}")
    md["salience_reason"] = "manual"
    md["weighed_at"] = at
    db.execute("UPDATE memory_item SET salience=?, metadata=?, updated=? WHERE id=?",
               (sal, json.dumps(md), at, args.id))
    db.commit()
    db.close()
    log_md(args.db, {"op": "weigh", "id": args.id, "salience": sal, "at": at},
           f"weigh {short(args.id)} {r['salience']}->{sal}")
    print(f"{short(args.id)} salience {r['salience']}->{sal}")
    return 0


def merge(db, keep, drops, at):
    """Fold each drop into keep: supersede it, move its user edges to keep,
    record its sources. Reversible (gc --restore DROP)."""
    kr = db.execute("SELECT metadata FROM memory_item WHERE id=?", (keep,)).fetchone()
    kmd = json.loads(kr["metadata"] or "{}")
    merged = kmd.setdefault("merged_from", [])
    q = ",".join("?" * len(SYSTEM_PREDICATES))
    for d in drops:
        if d == keep:
            raise ValueError("cannot merge an entry into itself")
        dmd = json.loads(db.execute("SELECT metadata FROM memory_item WHERE id=?",
                                    (d,)).fetchone()["metadata"] or "{}")
        merged.append({"id": d, "source": dmd.get("source", ""),
                       "provenance": dmd.get("provenance", "")})
        for e in db.execute(f"SELECT * FROM memory_edge WHERE (subject_id=? OR object_id=?)"
                            f" AND predicate NOT IN ({q})", (d, d, *SYSTEM_PREDICATES)).fetchall():
            a = keep if e["subject_id"] == d else e["subject_id"]
            b = keep if e["object_id"] == d else e["object_id"]
            db.execute("DELETE FROM memory_edge WHERE id=?", (e["id"],))
            if a != b:
                relate(db, a, b, e["predicate"], e["weight"], at)
        db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?", (keep, d))
        tombstone(db, d, f"merged into {keep}", at=at)
    db.execute("UPDATE memory_item SET metadata=?, updated=? WHERE id=?",
               (json.dumps(kmd), at, keep))


def cmd_merge(args):
    db = connect(args.db)
    try:
        keep = _live_ids(db, [resolve_id(db, args.keep)])[0]
        drops = _live_ids(db, [resolve_id(db, x) for x in args.drop.split(",") if x.strip()])
        at = getattr(args, "force_at", None) or now_iso()
        merge(db, keep, drops, at)
        db.commit()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        db.close()
    log_md(args.db, {"op": "merge", "keep": keep, "drop": drops, "at": at},
           f"merge {','.join(short(d) for d in drops)} -> {short(keep)}")
    print(f"merged {len(drops)} into {short(keep)}")
    return 0


def neighborhood(db, root, degrees):
    """Breadth-first walk over every edge (both directions, weight > 0) and
    every recorded conflict link, up to `degrees` hops. Returns
    {id: (hop, via_predicate)}."""
    seen = {root: (0, "")}
    frontier = [root]
    for hop in range(1, degrees + 1):
        nxt = []
        for n in frontier:
            pairs = [(e["object_id"], e["predicate"]) for e in db.execute(
                "SELECT object_id, predicate FROM memory_edge WHERE subject_id=? AND weight>0", (n,))]
            pairs += [(e["subject_id"], e["predicate"]) for e in db.execute(
                "SELECT subject_id, predicate FROM memory_edge WHERE object_id=? AND weight>0", (n,))]
            r = db.execute("SELECT metadata FROM memory_item WHERE id=?", (n,)).fetchone()
            md = json.loads(r["metadata"] or "{}") if r else {}
            pairs += [(c, "conflict") for c in md.get("possible_conflict_with", []) or []
                      if isinstance(c, str)]
            for m, pred in pairs:
                if m not in seen:
                    seen[m] = (hop, pred)
                    nxt.append(m)
        frontier = nxt
    return seen


def scan(db):
    """Quarantine candidates, by reason: conflict, duplicate, overlap."""
    rows = live_rows(db)
    ids = {r["id"] for r in rows}
    out = []
    for r in rows:
        md = json.loads(r["metadata"] or "{}")
        for c in md.get("possible_conflict_with", []) or []:
            if isinstance(c, str) and c in ids and r["id"] < c:
                out.append(("conflict", r["id"], c, 0.0))
    dup = set()
    for a, b, cos, _ in dedup_candidates(rows):
        out.append(("duplicate", a["id"], b["id"], cos))
        dup.add(frozenset((a["id"], b["id"])))
    try:
        import numpy as np
    except ImportError:
        return out
    vr = [(r, load_vec(r)) for r in rows]
    vr = [(r, v) for r, v in vr if v]
    if len(vr) > 1:
        m = np.asarray([v for _, v in vr], dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        sims = m @ m.T
        for i in range(len(vr)):
            for j in np.nonzero(sims[i, i + 1:] >= OVERLAP_COS)[0]:
                j = i + 1 + int(j)
                a, b = vr[i][0]["id"], vr[j][0]["id"]
                if frozenset((a, b)) not in dup:
                    out.append(("overlap", a, b, float(sims[i, j])))
    return out


def cmd_quarantine(args):
    """Quarantine an entry and, with --degrees N, everything within N hops of
    it (edges + conflict links). Dry run unless --apply. Reversible with
    --restore ROOT. --scan lists conflict/duplicate/overlap candidates."""
    db = connect(args.db)
    if args.scan:
        c = scan(db)
        for kind, a, b, v in c:
            print(f"{kind} {short(a)} {short(b)}" + (f" {v:.3f}" if v else ""))
        print(f"{len(c)} candidate pair(s)" if c else "no candidates")
        db.close()
        return 0
    at = getattr(args, "force_at", None) or now_iso()
    if args.restore:
        root = args.restore
        tag = f"quarantine via {root}"
        hit = [r["id"] for r in db.execute(
            "SELECT id, metadata FROM memory_item WHERE status='rejected'")
            if tag in (r["metadata"] or "")]
        n = sum(restore(db, i, at) for i in hit)
        db.commit()
        db.close()
        for i in hit:
            log_md(args.db, {"op": "restore", "id": i, "at": at}, f"restore {short(i)}")
        print(f"restored {n} row(s) quarantined via {short(root)}")
        return 0
    if not args.id:
        print("ERROR: give --id, --scan, or --restore", file=sys.stderr)
        return 2
    root = args.id
    if not db.execute("SELECT 1 FROM memory_item WHERE id=?", (root,)).fetchone():
        print(f"ERROR: no entry {root}", file=sys.stderr)
        return 2
    hood = neighborhood(db, root, args.degrees)
    plan, held = [], []
    for eid, (hop, pred) in sorted(hood.items(), key=lambda x: x[1][0]):
        r = db.execute("SELECT status, salience, metadata FROM memory_item WHERE id=?",
                       (eid,)).fetchone()
        if not r or r["status"] == "rejected":
            continue
        md = json.loads(r["metadata"] or "{}")
        if hop and is_exempt(r, md):
            held.append((eid, hop, pred))
        else:
            plan.append((eid, hop, pred))
    for eid, hop, pred in plan:
        print(f"{'quarantine' if args.apply else 'would quarantine'} {short(eid)} hop={hop}"
              + (f" via={pred}" if pred else ""))
    for eid, hop, pred in held:
        print(f"held (exempt) {short(eid)} hop={hop} via={pred}")
    if args.apply:
        reason = args.reason or "manual"
        for eid, hop, _ in plan:
            tombstone(db, eid, f"quarantine via {root} hop {hop}: {reason}", at=at)
        db.commit()
        for eid, hop, _ in plan:
            log_md(args.db, {"op": "quarantine", "id": eid, "at": at,
                             "reason": f"quarantine via {root} hop {hop}: {reason}"},
                   f"quarantine {short(eid)} (via {short(root)}, hop {hop})")
    elif plan:
        print("dry run — pass --apply")
    db.close()
    return 0


def _on_tmpfs(path):
    try:
        best, fs = "", ""
        for line in open("/proc/mounts"):
            parts = line.split()
            if len(parts) > 2 and path.startswith(parts[1]) and len(parts[1]) > len(best):
                best, fs = parts[1], parts[2]
        return fs in ("tmpfs", "ramfs")
    except OSError:
        return False


def cmd_doctor(args):
    """Check the runtime. One line per check; exit 1 on any FAIL."""
    res = []
    v = sys.version_info
    res.append(("ok" if v >= (3, 9) else "FAIL", "python", f"{v.major}.{v.minor}.{v.micro}"))
    res.append(("ok" if sqlite3.sqlite_version_info >= (3, 24) else "FAIL",
                "sqlite", sqlite3.sqlite_version))
    try:
        import numpy
        res.append(("ok", "numpy", numpy.__version__))
    except ImportError:
        res.append(("WARN", "numpy", "missing: vector search uses slow pure-Python path"))
    try:
        import fastembed
        res.append(("ok", "fastembed", getattr(fastembed, "__version__", "?")))
        fe = True
    except ImportError:
        res.append(("WARN", "fastembed", "missing: keyword-only search (install memorykit[vector])"))
        fe = False
    md = os.path.abspath(MODEL_DIR)
    cached = os.path.isdir(md) and any(os.scandir(md))
    if fe and getattr(args, "fetch_model", False):
        global _embedder_tried
        _embedder_tried = False
        ok = bool(embed_texts(["memorykit doctor"]))
        res.append(("ok" if ok else "FAIL", "model", f"{MODEL} loaded from {md}"
                    if ok else f"could not load {MODEL} (offline and not cached?)"))
        cached = ok
    elif fe:
        res.append(("ok" if cached else "WARN", "model",
                    f"cached in {md}" if cached else f"not cached in {md}; run doctor --fetch-model"))
    if cached and _on_tmpfs(md):
        res.append(("WARN", "model_dir", f"{md} is on tmpfs; it is wiped on reboot"))
    d = os.path.dirname(os.path.abspath(args.db))
    res.append(("ok" if os.access(d if os.path.isdir(d) else os.path.dirname(d), os.W_OK)
                else "FAIL", "db_dir", d))
    if os.path.exists(args.db):
        try:
            n = connect(args.db).execute("SELECT count(*) FROM memory_item").fetchone()[0]
            res.append(("ok", "db", f"{args.db} ({n} rows)"))
        except sqlite3.DatabaseError as e:
            res.append(("FAIL", "db", f"{args.db}: {e}"))
    res.append(("ok", "version", VERSION))
    for st, name, detail in res:
        print(f"{st:4} {name}: {detail}")
    return 1 if any(st == "FAIL" for st, _, _ in res) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.set_defaults(fn=lambda a: init_db(a.db))

    p = sub.add_parser("capture")
    p.add_argument("--speaker", required=True, choices=list(SPEAKER_PROVENANCE))
    p.add_argument("--text", required=True)
    p.add_argument("--session", default="")
    p.add_argument("--turn", default="")
    p.add_argument("--provenance", choices=list(PROVENANCE))
    p.add_argument("--salience", type=int)
    p.add_argument("--supersedes", default="")
    p.add_argument("--supersession-key", default="")
    p.add_argument("--parent-ids", default="")
    p.add_argument("--check", action="store_true")
    p.add_argument("--project", default="turns")
    p.add_argument("--protected", action="store_true",
                   help="exempt from GC and from cascade quarantine")
    p.add_argument("--source", default="",
                   help="who said it (a name or handle); kept apart from provenance")
    p.add_argument("--quote", default="",
                   help="text this turn quotes; stored as metadata, never indexed "
                        "as this speaker's words")
    p.add_argument("--quote-source", default="", help="who the quoted text is from")
    p.add_argument("--no-auto", action="store_true",
                   help="skip the settled-turn distill check for this capture")
    p.set_defaults(fn=cmd_capture)

    p = sub.add_parser("search")
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--chars", type=int, default=SNIPPET_CHARS,
                   help="max chars of each hit, picked around the query words")
    p.add_argument("--budget", type=int, default=0,
                   help="stop once ~N tokens of output are used (0 = no cap)")
    p.add_argument("--verbose", action="store_true",
                   help="show salience, confidence, and score")
    p.add_argument("--source", default="", help="only entries from this speaker")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("get")
    p.add_argument("--id", required=True)
    p.add_argument("--with-embedding", action="store_true")
    p.add_argument("--text", action="store_true", help="print only the entry's text")
    p.add_argument("--pretty", action="store_true")
    p.set_defaults(fn=cmd_get)

    p = sub.add_parser("supersede")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.set_defaults(fn=cmd_supersede)

    p = sub.add_parser("dedup")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=cmd_dedup)

    p = sub.add_parser("gc")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--restore", default="")
    p.add_argument("--eviction-report", action="store_true")
    p.set_defaults(fn=cmd_gc)

    p = sub.add_parser("feedback")
    p.add_argument("--id", required=True)
    p.add_argument("--rating", required=True, choices=["+1", "-1"])
    p.add_argument("--comment", default=None)
    p.set_defaults(fn=cmd_feedback)

    p = sub.add_parser("promote")
    p.add_argument("--id", required=True, help="the note to promote")
    p.add_argument("--evidence", required=True,
                   help="a live user-said or tool-observed entry")
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("reindex")
    p.add_argument("--md-dir", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--drop-missing", action="store_true",
                   help="with --force: rebuild even if DB rows absent from the log are lost (each is named)")
    p.set_defaults(fn=cmd_reindex)

    p = sub.add_parser("repair", help="heal DB/log drift without dropping rows (dry run by default)")
    p.add_argument("--drift", action="store_true", help="(default and only mode)")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--md-dir", default="")
    p.add_argument("--dedupe-log", action="store_true",
                   help="remove log records that are provable copies of a row "
                        "(same id, row exists, identical text); every id named")
    p.set_defaults(fn=cmd_repair)

    p = sub.add_parser("sleep",
                       help="manual catch-up distill (v5 does this a slice per turn; never required)")
    p.add_argument("--apply", action="store_true", help="default is a dry run")
    p.add_argument("--min-age-hours", type=float, default=0.0,
                   help="only consume turns at least this old")
    p.add_argument("--min-salience", type=int, default=SLEEP_MIN_SALIENCE)
    p.add_argument("--distiller", default="",
                   help="optional command: JSON turns on stdin, {\"text\"} out")
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("--settled-only", action="store_true",
                   help="only consume turns that have settled (see AUTO)")
    p.set_defaults(fn=cmd_sleep)

    p = sub.add_parser("bench", help="read-only recall check from [query, needle] pairs")
    p.add_argument("--queries", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--show-misses", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("health", help="read-only status; exit 1 if anything needs a look")
    p.add_argument("--pretty", action="store_true", help="every field, indented")
    p.set_defaults(fn=cmd_health)

    p = sub.add_parser("pre-turn", help="start of turn: repair, then recall within a budget")
    p.add_argument("--query", default="")
    p.add_argument("--budget", type=int, default=250)
    p.add_argument("--prefer-source", default="",
                   help="reserve up to 2 slots for this speaker (e.g. the user)")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--chars", type=int, default=220)
    p.add_argument("--repair", action="store_true",
                   help="run a drift repair this turn (else every MEMORYKIT_REPAIR_EVERY turns; 0 = never)")
    p.set_defaults(fn=cmd_pre_turn)

    p = sub.add_parser("post-turn", help="end of turn: journal, store, one upkeep step")
    p.add_argument("--jsonl", action="store_true",
                   help="read turns as JSON lines on stdin (batch; one model load)")
    p.add_argument("--speaker", default="user", choices=list(SPEAKER_PROVENANCE))
    p.add_argument("--text", default="")
    p.add_argument("--session", default="")
    p.add_argument("--turn", default="")
    p.add_argument("--source", default="")
    p.add_argument("--provenance", choices=list(PROVENANCE))
    p.add_argument("--salience", type=int)
    p.add_argument("--supersedes", default="")
    p.add_argument("--supersession-key", default="")
    p.add_argument("--parent-ids", default="")
    p.add_argument("--quote", default="")
    p.add_argument("--protected", action="store_true")
    p.add_argument("--no-upkeep", action="store_true")
    p.set_defaults(fn=cmd_post_turn)

    p = sub.add_parser("upkeep", help="run N bounded upkeep steps now (never required)")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--no-repair", action="store_true", help="skip the DB/log drift repair")
    p.set_defaults(fn=cmd_upkeep)

    p = sub.add_parser("upgrade", help="migrate an existing DB in place (backs up first)")
    p.add_argument("--no-backup", action="store_true")
    p.set_defaults(fn=cmd_upgrade)

    p = sub.add_parser("end-session", help="optional: let a session's turns settle now")
    p.add_argument("--session", required=True)
    p.set_defaults(fn=cmd_end_session)

    p = sub.add_parser("backfill")
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser("correlate")
    p.add_argument("--markdown", required=True)
    p.set_defaults(fn=cmd_correlate)

    p = sub.add_parser("relate", help="add/update/remove a weighted edge")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--predicate", default="related",
                   help="any word except derived_from/distilled_from (e.g. supports, contradicts)")
    p.add_argument("--weight", type=float, default=1.0, help="0 removes the edge")
    p.set_defaults(fn=cmd_relate)

    p = sub.add_parser("weigh", help="set a node's weight (salience 1-10)")
    p.add_argument("--id", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--salience", type=int)
    g.add_argument("--delta", type=int)
    p.set_defaults(fn=cmd_weigh)

    p = sub.add_parser("merge", help="fold entries into one (reversible)")
    p.add_argument("--keep", required=True)
    p.add_argument("--drop", required=True, help="id[,id...]")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("quarantine", help="quarantine by degrees of separation")
    p.add_argument("--id", default="")
    p.add_argument("--degrees", type=int, default=0)
    p.add_argument("--reason", default="")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--scan", action="store_true", help="list conflict/duplicate/overlap pairs")
    p.add_argument("--restore", default="", help="undo a quarantine by its root id")
    p.set_defaults(fn=cmd_quarantine)

    p = sub.add_parser("doctor", help="check python, sqlite, deps, model cache")
    p.add_argument("--fetch-model", action="store_true", help="download/load the model now")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("version")
    p.set_defaults(fn=lambda a: print(VERSION) or 0)

    args = ap.parse_args()
    if args.cmd not in ("init", "reindex", "doctor", "version", "upgrade",
                        "pre-turn", "post-turn") and not os.path.exists(args.db):
        print(f"no database at {args.db} — run `init` first", file=sys.stderr)
        return 1
    if not _REPLAYING and os.path.exists(args.db):
        try:
            _expand_ids(args)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
    if args.cmd in READ_ONLY_CMDS or args.cmd in ("init", "pre-turn", "post-turn", "upkeep"):
        return args.fn(args)  # read-only, or locks its own write sections
    with writer_lock(args.db):
        return args.fn(args)


READ_ONLY_CMDS = ("get", "health", "bench", "doctor", "version", "correlate")
SHORT_ID = int(os.environ.get("MEMORYKIT_SHORT_ID", "10"))


def short(eid):
    """Ids are printed short (10 hex chars by default) to save tokens; any
    unique prefix of 6+ chars is accepted wherever an id is taken."""
    return eid[:SHORT_ID] if SHORT_ID else eid


def resolve_id(db, ref):
    ref = (ref or "").strip()
    if not ref or len(ref) >= 24:
        return ref
    if len(ref) < 6:
        raise ValueError(f"id prefix {ref!r} too short (need 6+ chars)")
    m = [r[0] for r in db.execute(
        "SELECT id FROM memory_item WHERE id LIKE ? LIMIT 2", (ref + "%",))]
    if len(m) > 1:
        raise ValueError(f"id prefix {ref!r} is ambiguous; give more chars")
    return m[0] if m else ref


def _expand_ids(args):
    db = connect(args.db)
    try:
        for f in ("id", "old", "new", "restore", "keep", "a", "b"):
            if isinstance(getattr(args, f, None), str) and getattr(args, f):
                setattr(args, f, resolve_id(db, getattr(args, f)))
        for f in ("supersedes", "parent_ids", "drop"):
            v = getattr(args, f, None)
            if isinstance(v, str) and v:
                setattr(args, f, ",".join(resolve_id(db, x)
                                          for x in v.split(",") if x.strip()))
    except sqlite3.OperationalError:
        pass  # not initialized yet
    finally:
        db.close()


def _entry():
    """console_scripts entry point (pip install / pyproject)."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
