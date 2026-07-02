"""
run_eval.py

Replays each parsed trace's real, scripted user turns against a live
/chat endpoint and computes Recall@10 against the trace's labeled final
shortlist.

IMPORTANT CAVEAT (read this before trusting the numbers): this is a
literal replay of the exact user lines from the trace transcript, in
order — it does NOT adapt to whatever your agent actually asks, the way
the real grading harness's LLM-simulated user does. If your agent asks a
different clarifying question than the trace's agent did, the next
scripted line might not directly answer it, and your agent may end up
with less complete constraints than the trace shows. That's a limitation
of this quick local harness, not necessarily a bug in your agent — treat
low scores here as "investigate", not "definitely broken". It's still far
better than not testing against real traces at all.

Also respects the assignment's 8-turn cap: if a trace has more user turns
than fit in 8 total messages (this happens with C9, which has 7 user
turns), we stop feeding turns once the cap is hit and score whatever
shortlist the agent had committed to by then.

Usage:
    python3 run_eval.py --base-url http://localhost:8000
"""

import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path

from parse_traces import parse_all_traces, Trace
from retrieval import Catalog

MAX_TRANSCRIPT_TURNS = 8


def resolve_expected_urls(trace: Trace, catalog: Catalog) -> list[str]:
    """Map the trace's expected assessment NAMES to canonical catalog URLs,
    so scoring is robust to minor name-string differences (dashes, "(New)"
    suffixes, etc.) between the trace's table and the catalog JSON."""
    urls = []
    unresolved = []
    for name in trace.expected_names:
        item = catalog.get_by_name(name)
        if item:
            urls.append(item["url"])
        else:
            unresolved.append(name)
    if unresolved:
        print(f"  [warn] {trace.trace_id}: could not resolve to catalog: {unresolved}")
    return urls


def call_chat(base_url: str, messages: list[dict]) -> dict:
    req = urllib.request.Request(
        f"{base_url}/chat",
        data=json.dumps({"messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def run_trace(trace: Trace, base_url: str, verbose: bool = False) -> dict:
    messages: list[dict] = []
    last_recs: list[dict] = []
    hit_cap = False

    for i, user_msg in enumerate(trace.user_turns, 1):
        if len(messages) >= MAX_TRANSCRIPT_TURNS - 1:
            hit_cap = True
            if verbose:
                print(f"    [turn {i}] SKIPPED (would exceed turn cap): {user_msg[:60]!r}")
            break
        messages.append({"role": "user", "content": user_msg})
        if verbose:
            print(f"    [turn {i}] USER: {user_msg[:80]!r}")
        try:
            resp = call_chat(base_url, messages)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  [error] {trace.trace_id}: request failed: {e}")
            break
        if verbose:
            print(f"    [turn {i}] REPLY: {resp.get('reply', '')[:100]!r}")
            print(f"    [turn {i}] recs: {[r['name'] for r in resp.get('recommendations', [])]}")
            print(f"    [turn {i}] end_of_conversation: {resp.get('end_of_conversation')}")
        if resp.get("recommendations"):
            last_recs = resp["recommendations"]
        messages.append({"role": "assistant", "content": resp.get("reply", "")})
        if resp.get("end_of_conversation"):
            break
        if len(messages) >= MAX_TRANSCRIPT_TURNS:
            hit_cap = True
            break

    return {"recommendations": last_recs, "hit_cap": hit_cap, "turns_used": len(messages)}


def recall_at_10(predicted_urls: set[str], expected_urls: set[str]) -> float:
    if not expected_urls:
        return float("nan")
    return len(predicted_urls & expected_urls) / len(expected_urls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--traces-dir", default=str(Path(__file__).parent / "traces"))
    parser.add_argument("--verbose", action="store_true", help="print each turn's full response")
    parser.add_argument("--only", default=None, help="run a single trace id, e.g. C9")
    args = parser.parse_args()

    catalog = Catalog()
    traces = parse_all_traces(Path(args.traces_dir))
    if args.only:
        traces = [t for t in traces if t.trace_id == args.only]

    recalls = []
    print(f"{'trace':8} {'recall@10':>10}  {'turns':>6}  {'capped':>7}  details")
    print("-" * 90)
    for trace in traces:
        expected_urls = set(resolve_expected_urls(trace, catalog))
        if args.verbose:
            print(f"\n=== {trace.trace_id} (expected: {trace.expected_names}) ===")
        result = run_trace(trace, args.base_url, verbose=args.verbose)
        predicted_urls = {r["url"] for r in result["recommendations"]}
        score = recall_at_10(predicted_urls, expected_urls)
        recalls.append(score)

        missed = expected_urls - predicted_urls
        missed_names = [u.rstrip("/").rsplit("/", 1)[-1] for u in missed]
        cap_flag = "YES" if result["hit_cap"] else ""
        print(
            f"{trace.trace_id:8} {score:>10.2f}  {result['turns_used']:>6}  {cap_flag:>7}  "
            f"missed: {missed_names if missed_names else '-'}"
        )

    valid = [r for r in recalls if r == r]  # filter NaN
    mean_recall = sum(valid) / len(valid) if valid else float("nan")
    print("-" * 90)
    print(f"Mean Recall@10 across {len(valid)} traces: {mean_recall:.3f}")


if __name__ == "__main__":
    main()