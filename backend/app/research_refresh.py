"""Bounded refresh of tracked public evidence pages; curated claims remain versioned."""
import hashlib
import threading
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from urllib.request import Request, urlopen

from . import db

_lock = threading.Lock()


class PageTitle(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def fetch_source(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "CorsairResearch/0.1"})
    with urlopen(request, timeout=8) as response:
        content = response.read(1_000_001)
        if len(content) > 1_000_000:
            raise ValueError("Source exceeds the 1 MB refresh limit")
        if "html" not in response.headers.get("Content-Type", ""):
            raise ValueError("Day 1 refresh supports HTML evidence pages")
        parser = PageTitle()
        parser.feed(content.decode("utf-8", errors="replace"))
    return {"content_hash": hashlib.sha256(content).hexdigest(), "title": parser.title[:500]}


def refresh_sources(force: bool = False) -> dict:
    if not _lock.acquire(blocking=False):
        return {"status": "already_running", "checked": 0, "failed": 0}
    try:
        with db.connection() as conn:
            refresh_id = conn.execute("INSERT INTO research_refreshes (started_at) VALUES (?)", (db.now(),)).lastrowid
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        sources = db.fetch_all("SELECT * FROM research_sources WHERE (? OR checked_at IS NULL OR checked_at < ?) ORDER BY checked_at, url LIMIT 3", (force, cutoff))
        failed = 0
        for source in sources:
            try:
                metadata = fetch_source(source["url"])
                with db.connection() as conn:
                    conn.execute("UPDATE research_sources SET checked_at=?, succeeded_at=?, content_hash=?, title=?, error=NULL WHERE url=?",
                                 (db.now(), db.now(), metadata["content_hash"], metadata["title"], source["url"]))
            except Exception as exc:
                failed += 1
                with db.connection() as conn:
                    conn.execute("UPDATE research_sources SET checked_at=?, error=? WHERE url=?", (db.now(), str(exc)[:500], source["url"]))
        result = {"status": "completed", "checked": len(sources), "failed": failed}
        with db.connection() as conn:
            conn.execute("UPDATE research_refreshes SET finished_at=?, result_json=? WHERE id=?", (db.now(), db.dump(result), refresh_id))
        return result
    finally:
        _lock.release()


def refresh_status() -> dict:
    latest = db.fetch_one("SELECT * FROM research_refreshes ORDER BY id DESC LIMIT 1")
    return {"running": _lock.locked(), "latest": dict(latest) if latest else None,
            "sources": [dict(row) for row in db.fetch_all("SELECT * FROM research_sources ORDER BY url")]}
