"""Persistent case library, independent of project lifecycle tables."""
import hashlib
from urllib.parse import urlsplit, urlunsplit

from . import db
from .research_schema import ResearchCase


def initialize():
    with db.connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS research_cases (
                id TEXT PRIMARY KEY, identity TEXT NOT NULL UNIQUE,
                case_json TEXT NOT NULL, content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL,
                case_json TEXT NOT NULL, content_hash TEXT NOT NULL, recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_sources (
                url TEXT PRIMARY KEY, checked_at TEXT, succeeded_at TEXT,
                content_hash TEXT, title TEXT, error TEXT
            );
            CREATE TABLE IF NOT EXISTS research_refreshes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL,
                finished_at TEXT, result_json TEXT
            );
        """)


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def save_case(case: ResearchCase) -> dict:
    payload = case.model_dump(mode="json")
    identity = canonical_url(str(case.source_url)) + "|" + case.title.strip().casefold()
    case_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
    serialized = db.dump(payload)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    timestamp = db.now()
    with db.connection() as conn:
        previous = conn.execute("SELECT content_hash FROM research_cases WHERE identity = ?", (identity,)).fetchone()
        changed = previous is None or previous["content_hash"] != digest
        if changed:
            conn.execute("""INSERT INTO research_cases VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(identity) DO UPDATE SET case_json=excluded.case_json,
                content_hash=excluded.content_hash, updated_at=excluded.updated_at""",
                (case_id, identity, serialized, digest, timestamp, timestamp))
            conn.execute("INSERT INTO research_versions (case_id,case_json,content_hash,recorded_at) VALUES (?,?,?,?)",
                         (case_id, serialized, digest, timestamp))
        conn.execute("INSERT OR IGNORE INTO research_sources (url) VALUES (?)", (canonical_url(str(case.source_url)),))
    return {"id": case_id, "changed": changed}


def list_cases() -> list[dict]:
    return [_public(row) for row in db.fetch_all("SELECT * FROM research_cases ORDER BY id")]


def get_case(case_id: str) -> dict | None:
    row = db.fetch_one("SELECT * FROM research_cases WHERE id=?", (case_id,))
    return _public(row) if row else None


def _public(row) -> dict:
    return {"id": row["id"], **db.load(row["case_json"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}
