import json
import re
from collections import defaultdict


def normalize_discipline(sport: str, discipline: str) -> str:
    """
    Normalizes 'Men's canoe sprint K-2 1,000 metres' etc. so the same
    competition across different Olympic editions groups together,
    even with minor wording drift between years.
    """
    if not sport or not discipline:
        return None
    text = f"{sport}_{discipline}".lower()
    text = re.sub(r"[^\w\s]", "", text)       # strip punctuation
    text = re.sub(r"\s+", "_", text.strip())  # collapse whitespace
    return text


def build_preceded_by_edges(events_path: str, output_path: str):
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    groups = defaultdict(list)
    for e in events:
        if e.get("year") is None:
            continue
        key = normalize_discipline(e.get("sport"), e.get("discipline"))
        if key is None:
            continue
        groups[key].append(e)

    edges = []
    for key, group_events in groups.items():
        group_events.sort(key=lambda x: x["year"])
        for i in range(1, len(group_events)):
            edges.append({
                "event_id": group_events[i]["event_id"],
                "preceded_by_event_id": group_events[i - 1]["event_id"],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for edge in edges:
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")

    print(f"preceded_by edges created: {len(edges)}")
    print(f"Distinct competition groups found: {len(groups)}")
    print(f"Groups with only 1 edition (no edge possible): {sum(1 for g in groups.values() if len(g) == 1)}")


if __name__ == "__main__":
    build_preceded_by_edges("events_v2.json", "preceded_by_edges.json")