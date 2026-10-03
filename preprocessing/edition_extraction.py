import json

def build_editions(events_path: str, output_path: str):
    editions = {}

    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            year = event.get("year")
            season = event.get("season")
            if year is None or season is None:
                continue

            edition_id = f"{year}-{season.lower()}"
            if edition_id not in editions:
                editions[edition_id] = {
                    "edition_id": edition_id,
                    "year": year,
                    "season": season,
                    "host_city": None,  # not reliably extractable from infobox alone
                }

    with open(output_path, "w", encoding="utf-8") as f:
        for e in sorted(editions.values(), key=lambda x: (x["year"], x["season"])):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Editions written: {len(editions)}")


def add_edition_id_to_events(events_path: str, output_path: str):
    """Re-writes events.json adding edition_id so the edge can map cleanly."""
    updated = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            year, season = event.get("year"), event.get("season")
            event["edition_id"] = f"{year}-{season.lower()}" if year and season else None
            updated.append(event)

    with open(output_path, "w", encoding="utf-8") as f:
        for e in updated:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Events updated with edition_id: {len(updated)}")


if __name__ == "__main__":
    build_editions("events.json", "editions.json")
    add_edition_id_to_events("events.json", "events_v2.json")