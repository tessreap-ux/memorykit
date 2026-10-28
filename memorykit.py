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

Every capture and state change is also appended to ./memory-log/YYYY-MM-DD.md
(override with MEMORYKIT_MD_DIR), which is what `reindex` rebuilds from.
"""
import argparse
import datetime
import difflib
import json
import math
import os
import re
import sqlite3
import sys
import uuid

DEFAULT_DB = os.environ.get("MEMORYKIT_DB", "memory.db")
MODEL = os.environ.get("MEMORYKIT_MODEL", "BAAI/bge-small-en-v1.5")

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


def connect(db_path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    return db


def init_db(db_path):
    db = connect(db_path)
    db.executescript(open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "schema.sql")).read())
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


def novelty_adjust(db, text, salience):
    """Restatements can't inflate into importance; novelty gets +1.

    Compares against the top-8 live entries by embedding cosine (plus a
    difflib check, which catches near-verbatim restatements that
    mean-pooling under-scores when lengths differ). Never raises: on any
    failure the heuristic salience stands.
    """
    try:
        rows = live_rows(db)
        if not rows:
            return min(10, salience + 1), "novel"
        vec = None
        vecs = embed_texts([text[:4000]])
        if vecs:
            vec = vecs[0]
        scored = []
        for r in rows:
            rv = load_vec(r)
            c = cosine(vec, rv) if (vec and rv) else 0.0
            body = re.sub(r"^\[turn provenance=[^\]]*\]\n", "",
                          r["content"] or "")
            t = difflib.SequenceMatcher(None, text[:2000], body[:2000]).ratio()
            scored.append((max(c, 0.0), t))
        # Cosine and text similarity are judged independently across ALL
        # live rows. (Taking text similarity only from the top-8-by-cosine
        # rows breaks the gate when no embedder is installed: every cosine
        # is 0, so the top 8 are just the 8 oldest rows.)
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


def tombstone(db, eid, reason, cascade=False):
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
    md["tombstoned_at"] = now_iso()
    md["tombstone_reason"] = reason
    db.execute(
        "UPDATE memory_item SET status='rejected', metadata=?, updated=? "
        "WHERE id=?", (json.dumps(md), now_iso(), eid))
    children = db.execute(
        "SELECT subject_id FROM memory_edge "
        "WHERE object_id=? AND predicate='derived_from'", (eid,)).fetchall()
    for ch in children:
        tombstone(db, ch["subject_id"], f"cascade from {eid}", cascade=True)


# ---------------------------------------------------------------- commands

def cmd_capture(args):
    db = connect(args.db)
    text = args.text.strip()
    if len(text) < 10:
        print("SKIP: turn too short to index", file=sys.stderr)
        return 0
    if args.check:
        hits = search_rows(db, text, 5)
        print("=== possible related entries (review for contradiction) ===")
        for r, score, sim in hits:
            print(f"- id={r['id']} type={r['type']} conf={r['confidence']} "
                  f"sal={r['salience']} score={score:.3f}")
            print(f"  {(r['content'] or '')[:220]}".replace("\n", " "))
        print("=== end (advisory only — nothing auto-resolved) ===")
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
    if args.supersedes:
        metadata["supersedes"] = [s.strip() for s in args.supersedes.split(",") if s.strip()]
    if args.supersession_key:
        metadata["supersession_key"] = args.supersession_key.strip()
    parent_ids = [p.strip() for p in (args.parent_ids or "").split(",") if p.strip()]
    if parent_ids:
        metadata["parent_ids"] = parent_ids
    content = f"[turn provenance={provenance}]\n{text}"
    new_id = getattr(args, "force_id", None) or uuid.uuid4().hex[:24]
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
    for old_id in metadata.get("supersedes", []):
        db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?",
                   (new_id, old_id))
        tombstone(db, old_id, f"superseded by {new_id}")
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
            tombstone(db, p["id"], f"supersession key {skey} -> {new_id}")
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
    log_md(args.db, {
        "op": "capture", "id": new_id, "at": now, "speaker": args.speaker,
        "provenance": provenance, "text": text, "salience": salience,
        "salience_manual": bool(args.salience), "project": args.project,
        "session": args.session or "", "turn": args.turn,
        "supersedes": sorted(set(retired)), "supersession_key": skey or "",
        "parents": parent_ids, "protected": bool(args.protected),
        "novelty": novelty},
        f"[{provenance}] {text}")
    print(f"saved id={new_id} provenance={provenance} "
          f"salience={salience} confidence={confidence}")
    if vecs is None:
        print("note: no embedder installed — entry stored without a vector; "
              "run `backfill` after installing one "
              "(pip install fastembed)", file=sys.stderr)
    return 0


def search_rows(db, query, limit):
    """Top matches, live rows only. Reranks by similarity x salience x recency."""
    rows = live_rows(db)
    vecs = embed_texts([query[:4000]])
    qvec = vecs[0] if vecs else None
    qtok = set(t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2)
    now = datetime.datetime.now(datetime.timezone.utc)
    scored = []
    for r in rows:
        rv = load_vec(r)
        content = (r["content"] or "").lower()
        if qvec and rv:
            sim = cosine(qvec, rv)
        else:
            # No vectors (no embedder, or row not backfilled yet): fall back
            # to query-token overlap instead of full-string substring, which
            # is too strict to be useful ("photo folder name" should match
            # "the photo folder is called Quiet").
            ctok = set(re.findall(r"[a-z0-9]+", content))
            sim = len(qtok & ctok) / len(qtok) if qtok else 0.0
        try:
            accessed = datetime.datetime.fromisoformat(r["last_accessed_at"])
            age_days = max(0.0, (now - accessed).total_seconds() / 86400)
        except (ValueError, TypeError):
            age_days = 0.0
        recency = 1.0 / (1.0 + age_days / 30.0)
        score = sim * (r["salience"] / 10.0) * (0.5 + 0.5 * recency)
        if sim > 0:
            scored.append((r, score, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def cmd_search(args):
    db = connect(args.db)
    hits = search_rows(db, args.query, args.limit)
    for r, score, sim in hits:
        print(f"- id={r['id']} type={r['type']} sal={r['salience']} "
              f"conf={r['confidence']} score={score:.3f}")
        print(f"  {(r['content'] or '')[:400]}".replace("\n", " "))
        db.execute("UPDATE memory_item SET last_accessed_at=? WHERE id=?",
                   (now_iso(), r["id"]))
    db.commit()
    db.close()
    if not hits:
        print("(no matches)")


def cmd_get(args):
    db = connect(args.db)
    r = db.execute("SELECT * FROM memory_item WHERE id=?", (args.id,)).fetchone()
    db.close()
    if not r:
        print(f"no entry {args.id}")
        return 1
    print(json.dumps(dict(r), indent=2, default=str))
    return 0


def cmd_supersede(args):
    db = connect(args.db)
    db.execute("UPDATE memory_item SET superseded_by=? WHERE id=?",
               (args.new, args.old))
    tombstone(db, args.old, f"superseded by {args.new}")
    db.commit()
    db.close()
    log_md(args.db, {"op": "supersede", "old": args.old, "new": args.new},
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
            tombstone(db, drop, f"dedup merged into {keep}")
    if args.apply:
        db.commit()
        for keep, drop, _, _ in merges:
            log_md(args.db, {"op": "supersede", "old": drop, "new": keep,
                             "reason": "dedup"}, f"dedup {drop} -> {keep}")
        print(f"applied {len(merges)} merge(s); dropped rows kept as rejected")
    else:
        print("dry run — pass --apply to merge")
    db.close()
    return 0


def restore(db, eid):
    """Undo a quarantine and every cascade it caused. Returns rows restored."""
    db.execute("UPDATE memory_item SET status='confirmed', updated=? WHERE id=?",
               (now_iso(), eid))
    n = 1
    kids = db.execute(
        "SELECT subject_id FROM memory_edge WHERE object_id=? "
        "AND predicate='derived_from'", (eid,)).fetchall()
    for k in kids:
        r = db.execute("SELECT status, metadata FROM memory_item WHERE id=?",
                       (k["subject_id"],)).fetchone()
        if r and r["status"] == "rejected" and \
                f"cascade from {eid}" in (r["metadata"] or ""):
            n += restore(db, k["subject_id"])
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
    since_access = days_since("last_accessed_at")
    half = LOW_SALIENCE_HALF_LIFE if row["salience"] <= 3 else \
        HALF_LIFE.get(row["type"], 30)
    lam = math.log(2) / half
    r = 1.0 + 0.5 * math.exp(-since_access / 30.0)
    return u * math.exp(-lam * age) * r


def cmd_gc(args):
    db = connect(args.db)
    if args.restore:
        restored = restore(db, args.restore)
        db.commit()
        log_md(args.db, {"op": "restore", "id": args.restore},
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
    for row in live_rows(db):
        md = json.loads(row["metadata"]) if row["metadata"] else {}
        if is_exempt(row, md):
            continue  # exempt: high-importance rules and protected entries
        score = gc_score(row, feedback)
        if score < GC_NOISE_FLOOR:
            quarantined.append((row["id"], score))
            if args.apply:
                tombstone(db, row["id"], f"gc score {score:.4f} below floor")
    if args.apply:
        db.commit()
        for eid, score in quarantined:
            log_md(args.db, {"op": "quarantine", "id": eid,
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


def cmd_reindex(args):
    """Rebuild the database from the markdown log alone.

    Replays every mk record in timestamp order through the same code paths
    that wrote it, with logging off. Vectors are recomputed if an embedder
    is installed (otherwise run `backfill` later)."""
    global _REPLAYING
    d = args.md_dir or md_dir_for(args.db)
    files = sorted(f for f in os.listdir(d) if f.endswith(".md")) \
        if os.path.isdir(d) else []
    records = []
    for f in files:
        for n, line in enumerate(open(os.path.join(d, f), encoding="utf-8")):
            m = MK_RE.search(line)
            if m:
                try:
                    records.append((json.loads(m.group(1)), f, n))
                except json.JSONDecodeError:
                    print(f"skip malformed record {f}:{n + 1}", file=sys.stderr)
    records.sort(key=lambda x: (x[0].get("at", ""), x[1], x[2]))
    if os.path.exists(args.db):
        if not args.force:
            print(f"{args.db} exists; pass --force to replace it", file=sys.stderr)
            return 1
        os.remove(args.db)
    init_db(args.db)
    _REPLAYING = True
    counts = {}
    try:
        for rec, _, _ in records:
            op = rec.get("op")
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
                tombstone(db, rec["old"], f"superseded by {rec['new']}")
                db.commit()
                db.close()
            elif op == "quarantine":
                db = connect(args.db)
                tombstone(db, rec["id"], rec.get("reason", "quarantine"))
                db.commit()
                db.close()
            elif op == "restore":
                db = connect(args.db)
                restore(db, rec["id"])
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
    finally:
        _REPLAYING = False
    print(f"reindexed {len(records)} record(s) from {len(files)} file(s): "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


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
    p.set_defaults(fn=cmd_capture)

    p = sub.add_parser("search")
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("get")
    p.add_argument("--id", required=True)
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
    p.set_defaults(fn=cmd_reindex)

    p = sub.add_parser("backfill")
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser("correlate")
    p.add_argument("--markdown", required=True)
    p.set_defaults(fn=cmd_correlate)

    args = ap.parse_args()
    if args.cmd not in ("init", "reindex") and not os.path.exists(args.db):
        print(f"no database at {args.db} — run `init` first", file=sys.stderr)
        return 1
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
