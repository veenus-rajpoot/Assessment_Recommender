"""
retrieval.py

Retrieval over the cleaned SHL catalog.

Design choice: TF-IDF (scikit-learn) over the catalog's search_text, rather
than a hosted embedding API. Reasons:
  - 377 items is tiny. TF-IDF + cosine similarity is fast, deterministic,
    needs no API key/network call on the hot path, and is easy to defend
    in an interview ("why TF-IDF and not embeddings" has a real answer:
    at this corpus size and with short, keyword-dense descriptions,
    embeddings buy little and add latency + an external dependency inside
    the 30s per-call budget).
  - Job titles/skill names ("Java", "OPQ32r", ".NET") are exact-token
    sensitive. Embeddings tend to blur these; TF-IDF rewards exact matches.
  - It keeps the retrieval path testable without hitting any LLM/API.

The trade-off is TF-IDF won't bridge a big vocabulary gap (e.g. "leads a
team of engineers" -> "Manager" job level) as well as embeddings would.
We partially compensate with the constraint extractor in agent.py, which
pulls out explicit job-level / test-type / duration signals via the LLM and
applies them as a re-rank boost / hard filter on top of the TF-IDF score,
rather than relying on lexical similarity alone.
"""

from dataclasses import dataclass, field
from pathlib import Path
import difflib
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_PATH = Path(__file__).parent / "data" / "catalog_clean.json"


@dataclass
class Constraints:
    """Structured signal extracted from the conversation. All fields are
    optional soft/hard hints, not strict requirements — a user rarely gives
    complete info, and the evaluator's simulated user explicitly may say it
    has 'no preference' on some of these."""
    query_text: str = ""             # freeform role/skill description, used for TF-IDF
    test_types: list[str] = field(default_factory=list)   # e.g. ["K", "P"]
    job_levels: list[str] = field(default_factory=list)   # e.g. ["Mid-Professional"]
    max_duration_minutes: int | None = None
    remote_required: bool | None = None


class Catalog:
    def __init__(self, path: Path = CATALOG_PATH):
        self.items: list[dict] = json.loads(path.read_text())
        self._by_id = {item["id"]: item for item in self.items}
        self._by_name_lower = {item["name"].lower(): item for item in self.items}
        # Acronym index so "GSA" resolves to "Global Skills Assessment" even
        # though "GSA" never appears as a literal substring of the name.
        self._by_acronym = {}
        for item in self.items:
            words = re.findall(r"[A-Za-z]+", item["name"])
            acronym = "".join(w[0] for w in words if w[0].isupper()).lower()
            if len(acronym) >= 2:
                self._by_acronym.setdefault(acronym, []).append(item)

        corpus = [item["search_text"] for item in self.items]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

        # OPQ32r is SHL's flagship, general-purpose personality instrument —
        # several of the provided conversation traces show the reference
        # agent explicitly defaulting to it whenever a personality signal is
        # requested and no more specific instrument (e.g. a safety- or
        # sales-specific one) clearly fits better. Structurally, the catalog
        # also contains many derivative "OPQ ... Report" / "... Leadership
        # Report" products that are administratively run off the same
        # OPQ32r data but are separate catalog entries whose descriptions
        # repeat domain keywords (e.g. "leadership") heavily enough to
        # out-rank the base instrument on pure TF-IDF, even though OPQ32r is
        # what should actually be selected. We give it a small, explicit
        # boost to correct for that, calibrated so it doesn't override a
        # genuinely more specific match on its own topic (verified against
        # the Dependability & Safety Instrument case, which still ranks
        # ahead of OPQ32r for safety-specific queries after this boost).
        self._flagship_boost = {
            item["id"]: 0.3
            for item in self.items
            if item["name"] == "Occupational Personality Questionnaire OPQ32r"
        }

    def get_by_name(self, name: str) -> dict | None:
        """Exact-ish lookup used by the 'compare' behavior (and by the eval
        harness to resolve trace ground-truth names), so comparisons are
        grounded in the catalog record rather than the model's memory."""
        name_l = name.lower().strip()
        if name_l in self._by_name_lower:
            return self._by_name_lower[name_l]
        # acronym match, e.g. "GSA" -> "Global Skills Assessment"
        if name_l in self._by_acronym and len(self._by_acronym[name_l]) == 1:
            return self._by_acronym[name_l][0]
        # fall back to substring match (e.g. user says "OPQ" for "OPQ32r")
        candidates = [
            item for item in self.items if name_l in item["name"].lower()
        ]
        if len(candidates) == 1:
            return candidates[0]

        # Fuzzy fallback: normalize away punctuation/spacing differences
        # (e.g. "SVAR Spoken English (US)" vs catalog's "SVAR - Spoken
        # English (US)"), then fall back to closest string match. This
        # matters most for resolving trace/ground-truth names that are
        # trivial variants of the real catalog name, not different items.
        def normalize(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        name_norm = normalize(name)
        norm_matches = [
            item for item in self.items if normalize(item["name"]) == name_norm
        ]
        if len(norm_matches) == 1:
            return norm_matches[0]

        close = difflib.get_close_matches(
            name_l, [item["name"].lower() for item in self.items], n=1, cutoff=0.8
        )
        if close:
            return self._by_name_lower.get(close[0])

        return None

    def search(self, constraints: Constraints, top_k: int = 10) -> list[dict]:
        query = constraints.query_text.strip()
        if not query:
            scores = [0.0] * len(self.items)
        else:
            q_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self.matrix)[0]

        scored = []
        for item, score in zip(self.items, scores):
            score = float(score)

            # Hard filters: drop items that violate an explicit constraint
            # rather than just down-ranking them. A user who says "under 20
            # minutes" doesn't want a 45-minute test ranked #1 because its
            # text matched well.
            if constraints.max_duration_minutes is not None:
                dur = item["duration_minutes"]
                if dur is not None and dur > constraints.max_duration_minutes:
                    continue
            if constraints.remote_required and not item["remote"]:
                continue

            # Soft boosts: nudge score for matching test_type / job_level
            # rather than filtering, since these are often only loosely
            # implied by the conversation. The type boost is deliberately
            # large (0.5, vs typical TF-IDF cosine scores of 0.01-0.15 on
            # this corpus) so an EXPLICITLY requested type reliably beats
            # incidental keyword noise — e.g. without this, a generic
            # "Entry Level X Solution" bundle that coincidentally shares a
            # token with the query can outrank a dedicated, correctly-typed
            # assessment (like OPQ32r for a "personality" request) purely on
            # noise, even though the type match is the real, stated signal.
            if constraints.test_types:
                overlap = set(constraints.test_types) & set(item["test_types"])
                if overlap:
                    score += 0.5 * len(overlap)
                    score -= 0.05 * (len(item["test_types"]) - 1)
                    score += self._flagship_boost.get(item["id"], 0.0)
                elif score == 0.0:
                    continue  # no text match AND no type match -> not relevant
            if constraints.job_levels:
                overlap = set(constraints.job_levels) & set(item["job_levels"])
                if overlap:
                    score += 0.15 * len(overlap)

            if score > 0:
                scored.append((score, item))

        # Tie-break ties on (a) score descending, then (b) specialization —
        # fewer total test_types on the item ranks first. Without this,
        # broad multi-type bundle products (e.g. "Entry Level X Solution",
        # tagged with 5+ types) tie with dedicated single-type assessments
        # (e.g. OPQ32r, tagged only "P") on the flat type-match boost, and
        # ties resolve to arbitrary catalog order instead of favoring the
        # more clearly-on-target dedicated assessment.
        scored.sort(key=lambda pair: (-pair[0], len(pair[1]["test_types"])))

        # If the user's constraints span multiple test_types (e.g. K + P for
        # "Java developer who works with stakeholders"), don't just take a
        # single global top-N by score. Pure TF-IDF text similarity will
        # almost always favor whichever type has more literal keyword
        # overlap (technical skill names are keyword-dense; personality
        # items are not), which would silently crowd out an entire relevant
        # category even though the type-match boost applied above. Instead,
        # round-robin across the requested types so each gets fair
        # representation, then fill any remaining slots by pure score.
        if len(constraints.test_types) > 1:
            by_type: dict[str, list[dict]] = {t: [] for t in constraints.test_types}
            leftover: list[dict] = []
            for score, item in scored:
                matched_types = [t for t in constraints.test_types if t in item["test_types"]]
                if matched_types:
                    by_type[matched_types[0]].append(item)
                else:
                    leftover.append(item)

            result: list[dict] = []
            seen_ids = set()
            round_idx = 0
            type_list = constraints.test_types
            while len(result) < top_k and any(by_type[t] for t in type_list):
                t = type_list[round_idx % len(type_list)]
                if by_type[t]:
                    candidate = by_type[t].pop(0)
                    if candidate["id"] not in seen_ids:
                        result.append(candidate)
                        seen_ids.add(candidate["id"])
                round_idx += 1
            for item in leftover:
                if len(result) >= top_k:
                    break
                if item["id"] not in seen_ids:
                    result.append(item)
                    seen_ids.add(item["id"])
            return result[:top_k]

        return [item for _, item in scored[:top_k]]