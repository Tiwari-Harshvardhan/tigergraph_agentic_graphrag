import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.tigergraph_connection import get_connection


conn = get_connection()

queries = conn.getInstalledQueries()

print("Installed queries:")
for query in queries:
    print(query)