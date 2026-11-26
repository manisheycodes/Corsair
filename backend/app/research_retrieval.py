"""Deterministic lexical retrieval baseline; result scores are relevance, not confidence."""
import re

from .research_store import list_cases

STOP_WORDS = {"the", "a", "an", "to", "of", "and", "for", "in", "with", "is", "i", "want", "how"}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


def search_cases(query: str, task_type: str | None = None, limit: int = 5) -> list[dict]:
    terms = tokens(query)
    ranked = []
    if not terms:
        return []
    for case in list_cases():
        if task_type and case["task_type"] != task_type:
            continue
        strong = tokens(" ".join([case["title"], case["domain"], *case["models"]]))
        body = tokens(" ".join([case["situation"], case["dataset_name"] or "", *case["features"]]))
        matched = terms & (strong | body)
        score = len(terms & strong) * 2 + len(terms & body)
        if score:
            ranked.append({"case": case, "relevance_score": score, "matched_terms": sorted(matched)})
    return sorted(ranked, key=lambda item: (-item["relevance_score"], item["case"]["id"]))[:limit]
