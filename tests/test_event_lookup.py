import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.tigergraph_connection import get_connection


def main():
    conn = get_connection()

    event_id = "Q303623"

    result = conn.runInstalledQuery(
        "event_lookup",
        {"event_id": event_id},
    )

    print("RAW TIGERGRAPH RESPONSE")
    print("=" * 80)
    print(result)

    print("\nPARSED RESPONSE")
    print("=" * 80)

    for index, item in enumerate(result):
        print(f"\nRESULT {index}")
        print(item)


if __name__ == "__main__":
    main()