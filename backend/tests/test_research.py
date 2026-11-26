import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import db
from app.main import app
from app.research_schema import ResearchCase
from app import research_refresh as refresh
from app.research_store import initialize, list_cases, save_case, get_case, case_versions
from app.research_seeds import seed_cases
from app.research_retrieval import search_cases


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATABASE_PATH", tmp_path / "research.db")
    db.initialize()
    initialize()
    seed_cases()


def test_persistence_idempotence_and_revision_history(library):
    assert len(list_cases()) == 3
    assert seed_cases()["changed"] == 0
    original = list_cases()[0]
    case = ResearchCase(**original)
    assert save_case(case)["changed"] is False
    case.situation += " Additional reviewed context."
    assert save_case(case)["changed"] is True
    initialize()
    assert len(list_cases()) == 3
    assert len(case_versions(original["id"])) == 2
    seed_cases()
    assert get_case(original["id"])["situation"].endswith("Additional reviewed context.")


def test_fragment_urls_deduplicate(library):
    original = list_cases()[0]
    case = ResearchCase(**{**original, "source_url": original["source_url"] + "#section"})
    assert save_case(case)["id"] == original["id"]
    assert len(list_cases()) == 3


def test_claims_require_evidence_and_missing_details_stay_unknown(library):
    original = list_cases()[0]
    assert "dataset_license" in original["unknowns"]
    with pytest.raises(ValidationError, match="model-choice"):
        ResearchCase(**{**original, "model_reason": "A fabricated reason"})
    with pytest.raises(ValidationError, match="situation and models"):
        ResearchCase(**{**original, "evidence": [original["evidence"][0]]})


def test_retrieval_filters_and_cites(library):
    results = search_cases("handwritten digits K means", "clustering")
    assert len(results) == 1
    assert results[0]["case"]["source_url"].startswith("https://")
    assert results[0]["matched_terms"]
    assert search_cases("completelyunrelatedword") == []
    assert search_cases("the and of") == []


def test_refresh_records_failures_and_respects_due_interval(library, monkeypatch):
    calls = []

    def offline(url):
        calls.append(url)
        raise OSError("offline fixture")

    monkeypatch.setattr(refresh, "fetch_source", offline)
    assert refresh.refresh_sources() == {"status": "completed", "checked": 3, "failed": 3}
    assert refresh.refresh_sources()["checked"] == 0
    assert len(calls) == 3
    assert all(source["error"] for source in refresh.refresh_status()["sources"])
    monkeypatch.setattr(refresh, "fetch_source", lambda url: {"title": "Recovered page", "content_hash": "abc"})
    assert refresh.refresh_sources(force=True)["failed"] == 0
    assert all(source["error"] is None and source["succeeded_at"] for source in refresh.refresh_status()["sources"])
    assert len(list_cases()) == 3


def test_library_api_and_offline_startup(library, monkeypatch):
    monkeypatch.setenv("AUTOBUILD_RESEARCH_REFRESH", "false")
    headers = {"X-API-Key": "local-development-key"}
    with TestClient(app) as client:
        assert client.get("/research/cases").status_code == 401
        cases = client.get("/research/cases", headers=headers).json()
        assert len(cases) == 3
        response = client.get("/research/search?q=digits&task_type=classification", headers=headers)
        assert len(response.json()["results"]) == 1
        assert client.get("/research/cases/unknown", headers=headers).status_code == 404
        assert len(client.get("/research/cases/" + cases[0]["id"], headers=headers).json()["versions"]) == 1
        assert client.post("/research/cases", headers=headers, json=[cases[0]]).json()["changed"] == 0
        monkeypatch.setattr(refresh, "fetch_source", lambda url: {"title": "Fixture", "content_hash": "abc"})
        assert client.post("/research/refresh", headers=headers, json={"force": True}).status_code == 202
        assert client.get("/research/refresh", headers=headers).json()["latest"]["finished_at"]
