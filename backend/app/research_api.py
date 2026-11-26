from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from .research_schema import ResearchCase, RefreshRequest
from .research_store import case_versions, get_case, ingest_cases, list_cases
from .research_retrieval import search_cases
from .research_refresh import refresh_sources, refresh_status

router = APIRouter(prefix="/research", tags=["Research library"])


@router.get("/cases")
def cases():
    return list_cases()


@router.post("/cases")
def ingest(records: list[ResearchCase]):
    return ingest_cases(records)


@router.get("/search")
def search(q: str = Query(min_length=1, max_length=1000), task_type: str | None = None,
           limit: int = Query(default=5, ge=1, le=20)):
    return {"method": "lexical", "results": search_cases(q, task_type, limit)}


@router.get("/cases/{case_id}")
def detail(case_id: str):
    case = get_case(case_id)
    if case is None:
        raise HTTPException(404, "Research case not found")
    return {"case": case, "versions": case_versions(case_id)}


@router.post("/refresh", status_code=202)
def refresh(payload: RefreshRequest, tasks: BackgroundTasks):
    tasks.add_task(refresh_sources, payload.force)
    return {"status": "accepted", "status_endpoint": "/research/refresh"}


@router.get("/refresh")
def status():
    return refresh_status()
