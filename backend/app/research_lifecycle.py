"""Offline seed initialization and optional periodic background source checks."""
import os
import threading
from contextlib import asynccontextmanager

from .research_store import initialize
from .research_seeds import seed_cases
from .research_refresh import refresh_sources


@asynccontextmanager
async def research_lifespan(app):
    initialize()
    seed_cases()
    stopped = threading.Event()

    def refresh_loop():
        while not stopped.is_set():
            refresh_sources()
            stopped.wait(3600)

    enabled = os.getenv("AUTOBUILD_RESEARCH_REFRESH", "true").lower() == "true"
    if enabled:
        threading.Thread(target=refresh_loop, name="research-refresh", daemon=True).start()
    try:
        yield
    finally:
        stopped.set()
