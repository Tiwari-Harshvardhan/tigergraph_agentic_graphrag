import json
import re
from pathlib import Path


def get_infobox_type(text: str) -> str | None:
    m = re.match(r"^\[[Ii]nfobox ([^\]]+)\]", text)
    return m.group(1) if m else None


def extract_infobox(text: str) -> dict:
    lines = text.split("\n")
    fields = {}
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            break
        m = re.match(r"^(\w+):\s*(.*)$", stripped)
        if not m:
            break
        fields[m.group(1)] = m.group(2).strip()
    return fields


def parse_games(games_value: str) -> dict:
    m = re.match(r"^(\d{4})\s+(Summer|Winter)$", games_value.strip())
    if not m:
        return {"year": None, "season": None}
    return {"year": int(m.group(1)), "season": m.group(2)}


def parse_sport_from_title(title: str) -> str:
    m = re.match(r"^(.+?) at the \d{4} (?:Summer|Winter) Olympics", title.strip())
    return m.group(1).strip() if m else None


def extract_event_fields(doc: dict, infobox: dict) -> dict:
    games = parse_games(infobox.get("games", ""))
    competitor_count = int(infobox["competitors"]) if infobox.get("competitors", "").isdigit() else None
    nation_count = int(infobox["nations"]) if infobox.get("nations", "").isdigit() else None
    date = infobox.get("date") or infobox.get("dates")

    return {
        "event_id": doc["doc_id"],
        "doc_id": doc["doc_id"],
        "sport": parse_sport_from_title(doc.get("title", "")),
        "edition": f"{games['year']} {games['season']} Olympics" if games["year"] else None,
        "year": games["year"],
        "season": games["season"],
        "discipline": infobox.get("event"),
        "venue": infobox.get("venue"),
        "date": date,
        "competitor_count": competitor_count,
        "nation_count": nation_count,
        "gold_medalist": infobox.get("gold"),
        "gold_noc": infobox.get("goldNOC"),
    }


def process_corpus(input_path: str, documents_out: str, events_out: str):
    documents = []
    events = []
    failures = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text = doc.get("text", "")
            infobox_type = get_infobox_type(text)

            # EVERY doc gets a full Document record
            documents.append({
                "doc_id": doc["doc_id"],
                "title": doc.get("title"),
                "url": doc.get("url"),
                "approx_tokens": doc.get("approx_tokens"),
                "text": doc.get("text"),
                "wikidata_qid": doc.get("wikidata_qid"),
                "wikipedia_pageid": doc.get("wikipedia_pageid"),
                "infobox_type": infobox_type,
            })

            # ONLY Olympic-event docs also get an Event record
            if infobox_type == "Olympic event":
                infobox = extract_infobox(text)
                event = extract_event_fields(doc, infobox)
                events.append(event)

                core_fields = ["sport", "edition", "venue", "date", "competitor_count", "nation_count"]
                missing = [k for k in core_fields if event.get(k) is None]
                if missing:
                    failures.append({"doc_id": doc["doc_id"], "title": doc.get("title"), "missing": missing})

    with open(documents_out, "w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(events_out, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Documents written: {len(documents)}  (should be 2951)")
    print(f"Events written: {len(events)}  (should be 2162)")
    print(f"Events with missing core field: {len(failures)}")
    if failures:
        failures_path = Path(events_out).with_name("extraction_failures.json")
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    corpus_dir = Path(__file__).resolve().parent
    process_corpus(
        corpus_dir / "corpus.jsonl",
        corpus_dir / "documents.json",
        corpus_dir / "events.json",
    )