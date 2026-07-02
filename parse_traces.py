"""
parse_traces.py

Parses the provided markdown conversation traces (C1.md .. C10.md) into a
structured form the eval harness can drive:
  - user_turns: the ordered list of the user's scripted lines
  - expected: the labeled expected shortlist, taken from the LAST turn's
    table (the one marked `end_of_conversation: **true**`) — this is the
    ground truth we score Recall@10 against.

Format assumptions, based on inspecting all 10 provided traces:
  - Turns are delimited by "### Turn N" headers.
  - Within a turn, "**User**" is followed by a blockquoted message (lines
    starting with ">"), possibly multi-line/multi-paragraph.
  - The agent's final answer (when end_of_conversation is true) contains a
    markdown table with a "Name" column — that's our expected shortlist.
  - A few traces have turns with no table (pure clarifying questions or
    refusals) — those are skipped when looking for the final table, we
    specifically want the LAST table in the document, not the last turn's
    text if it has no table.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Trace:
    trace_id: str
    user_turns: list[str]
    expected_names: list[str]


def _extract_user_turns(text: str) -> list[str]:
    """Pull each turn's blockquoted user message, in order."""
    turns = []
    # Split on turn headers, then find the User block within each chunk.
    chunks = re.split(r"### Turn \d+", text)[1:]  # [0] is preamble before Turn 1
    for chunk in chunks:
        user_match = re.search(
            r"\*\*User\*\*\s*\n((?:>.*\n?)+)", chunk
        )
        if not user_match:
            continue
        raw_lines = user_match.group(1).splitlines()
        # Strip leading "> " (or bare ">") from each blockquote line and
        # rejoin, preserving paragraph breaks.
        cleaned = []
        for line in raw_lines:
            line = line.strip()
            if line.startswith(">"):
                line = line[1:].strip()
            cleaned.append(line)
        message = "\n".join(cleaned).strip()
        if message:
            turns.append(message)
    return turns


def _extract_final_shortlist(text: str) -> list[str]:
    """Find the LAST markdown table in the document (that's the final,
    end_of_conversation:true shortlist) and pull the Name column."""
    # Markdown tables: a header row containing "Name", a separator row of
    # dashes, then data rows. Find all such tables, take the last.
    table_blocks = re.findall(
        r"(\|[^\n]*Name[^\n]*\|\n\|[-\s|]+\|\n(?:\|.*\|\n?)+)", text
    )
    if not table_blocks:
        return []
    last_table = table_blocks[-1]
    rows = last_table.strip().splitlines()
    header = [c.strip() for c in rows[0].split("|")]
    name_col_idx = next(i for i, c in enumerate(header) if c.lower() == "name")

    names = []
    for row in rows[2:]:  # skip header + separator
        cols = [c.strip() for c in row.split("|")]
        if len(cols) <= name_col_idx:
            continue
        name = cols[name_col_idx]
        if name:
            names.append(name)
    return names


def parse_trace_file(path: Path) -> Trace:
    text = path.read_text()
    return Trace(
        trace_id=path.stem,
        user_turns=_extract_user_turns(text),
        expected_names=_extract_final_shortlist(text),
    )


def parse_all_traces(dir_path: Path) -> list[Trace]:
    return [parse_trace_file(p) for p in sorted(dir_path.glob("*.md"))]


if __name__ == "__main__":
    traces_dir = Path(__file__).parent / "traces"
    traces = parse_all_traces(traces_dir)
    for t in traces:
        print(f"=== {t.trace_id} ===")
        print(f"  {len(t.user_turns)} user turns")
        for i, turn in enumerate(t.user_turns, 1):
            preview = turn.replace("\n", " ")[:70]
            print(f"    [{i}] {preview}")
        print(f"  expected ({len(t.expected_names)}): {t.expected_names}")
        print()