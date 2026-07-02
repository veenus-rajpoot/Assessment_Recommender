"""
Offline sanity checks for retrieval.py — no LLM, no network, runs anywhere.
This does NOT test agent.py/main.py (those need the LLM + FastAPI/pydantic
installed); it only proves the retrieval core is sane before you spend LLM
calls debugging the rest of the pipeline.

Run: python3 tests/test_retrieval_offline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import Catalog, Constraints  # noqa: E402


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main():
    catalog = Catalog()
    all_ok = True

    # 1. Basic role query should surface Java-relevant items near the top.
    results = catalog.search(Constraints(query_text="Java developer who works with stakeholders"), top_k=5)
    names = [r["name"] for r in results]
    all_ok &= check(
        f"Java query surfaces a Java item in top 5 (got: {names})",
        any("java" in n.lower() for n in names),
    )

    # 2. Duration hard filter is respected.
    results = catalog.search(
        Constraints(query_text="administrative assistant", max_duration_minutes=15), top_k=10
    )
    all_ok &= check(
        "max_duration_minutes filter excludes longer items",
        all(r["duration_minutes"] is None or r["duration_minutes"] <= 15 for r in results),
    )

    # 3. test_type boost: asking for personality tests should rank P-type items up.
    results = catalog.search(
        Constraints(query_text="sales manager", test_types=["P"]), top_k=10
    )
    p_count = sum(1 for r in results if "P" in r["test_types"])
    all_ok &= check(
        f"test_type=['P'] boost yields personality items in results ({p_count}/{len(results)})",
        p_count > 0,
    )

    # 4. Acronym-based compare lookup.
    gsa = catalog.get_by_name("GSA")
    all_ok &= check("acronym lookup resolves GSA", gsa is not None and "Global Skills" in gsa["name"])

    opq = catalog.get_by_name("OPQ32r")
    all_ok &= check("exact-ish name lookup resolves OPQ32r", opq is not None and "OPQ32r" in opq["name"])

    # 5. Schema sanity: every catalog item has a URL from the real domain.
    bad_urls = [item for item in catalog.items if "shl.com" not in item["url"]]
    all_ok &= check("all catalog URLs are on shl.com", len(bad_urls) == 0)

    print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
