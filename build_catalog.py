"""
build_catalog.py

Cleans the raw scraped SHL catalog (shl_product_catalog.json) into a normalized
structure that the retrieval layer consumes. Run once at build/deploy time
(or on service startup) to produce data/catalog_clean.json.

Why this exists as a separate step instead of parsing raw JSON inline in the
retrieval layer: the raw scrape has inconsistent/duplicate fields (job_levels
vs job_levels_raw), occasional bad control characters, and category labels
("keys") that need mapping onto SHL's standard single-letter test_type codes
that the API response schema expects (e.g. "K", "P"). Doing that once here
means retrieval.py never has to think about scrape artifacts.
"""

import json
import re
from pathlib import Path

RAW_PATH = Path(__file__).parent / "data" / "shl_product_catalog.json"
OUT_PATH = Path(__file__).parent / "data" / "catalog_clean.json"

# SHL's standard test-type taxonomy. The scraped "keys" field uses the long
# form category names; the API schema (per the assignment spec) wants the
# short code, e.g. {"name": "OPQ32r", "test_type": "P"}.
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def parse_duration_minutes(duration_raw: str, duration: str) -> int | None:
    """Extract an integer minute count from either duration field. Returns
    None if no duration is listed (some catalog entries omit it entirely -
    we should not silently treat that as 0, since 0 would wrongly pass any
    'under N minutes' filter)."""
    for source in (duration_raw, duration):
        if not source:
            continue
        match = re.search(r"(\d+)", source)
        if match:
            return int(match.group(1))
    return None


def clean_record(raw: dict) -> dict | None:
    # Skip entries the scraper itself flagged as broken.
    if raw.get("status") != "ok":
        return None

    name = (raw.get("name") or "").strip()
    link = (raw.get("link") or "").strip()
    if not name or not link:
        return None

    test_types = sorted({
        CATEGORY_TO_CODE[k] for k in raw.get("keys", []) if k in CATEGORY_TO_CODE
    })

    description = (raw.get("description") or "").strip()
    job_levels = raw.get("job_levels") or []
    languages = raw.get("languages") or []
    duration_minutes = parse_duration_minutes(
        raw.get("duration_raw", ""), raw.get("duration", "")
    )

    # search_text is what gets embedded / indexed. Deliberately repeats the
    # name (weights it higher for keyword/TF-IDF matching) and spells out
    # test types and job levels in plain words, since users say "personality
    # test" or "entry level", not "P" or "Graduate".
    search_text_parts = [
        name, name,  # repeated for weighting
        description,
        "Job levels: " + ", ".join(job_levels) if job_levels else "",
        "Test type: " + ", ".join(test_types) if test_types else "",
    ]
    search_text = " ".join(p for p in search_text_parts if p)

    return {
        "id": raw.get("entity_id"),
        "name": name,
        "url": link,
        "description": description,
        "test_types": test_types,
        "job_levels": job_levels,
        "languages": languages,
        "duration_minutes": duration_minutes,
        "remote": raw.get("remote") == "yes",
        "adaptive": raw.get("adaptive") == "yes",
        "search_text": search_text,
    }


def build():
    # strict=False: the raw scrape has a handful of literal control characters
    # inside string values (line breaks copy-pasted from the source page),
    # which the default strict JSON parser rejects.
    raw_items = json.loads(RAW_PATH.read_text(errors="replace"), strict=False)
    cleaned = []
    seen_ids = set()
    for raw in raw_items:
        rec = clean_record(raw)
        if rec is None:
            continue
        if rec["id"] in seen_ids:
            continue  # dedupe defensively; scrapes sometimes double-list items
        seen_ids.add(rec["id"])
        cleaned.append(rec)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(cleaned, indent=2))
    print(f"Cleaned {len(cleaned)} / {len(raw_items)} catalog items -> {OUT_PATH}")

    no_type = sum(1 for r in cleaned if not r["test_types"])
    no_duration = sum(1 for r in cleaned if r["duration_minutes"] is None)
    print(f"  items with no test_type mapped: {no_type}")
    print(f"  items with no parsable duration: {no_duration}")


if __name__ == "__main__":
    build()
