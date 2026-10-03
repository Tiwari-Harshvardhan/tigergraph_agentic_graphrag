import os
from pathlib import Path

import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

def main():
    input_path = Path("corpus/chunks_with_embeddings_v2.json")
    query_name = "vector_search"
    embedding_dimension = 384

    with input_path.open(encoding="utf-8") as input_file:
        header = input_file.readline().rstrip("\r\n")
        if header != "chunk_id|embedding":
            raise ValueError(f"Unexpected input header: {header!r}")

        first_record = input_file.readline().rstrip("\r\n")

    fields = first_record.split("|", 1)
    if len(fields) != 2:
        raise ValueError("The first embedding record is not pipe-delimited")

    chunk_id, embedding_text = fields
    embedding = [float(value) for value in embedding_text.split(",")]
    if len(embedding) != embedding_dimension:
        raise ValueError(
            f"Expected {embedding_dimension} values, found {len(embedding)}"
        )

    host = os.getenv("TG_HOST")
    graphname = os.getenv("TG_GRAPHNAME")
    secret = os.getenv("TG_SECRET")
    if not all((host, graphname, secret)):
        raise RuntimeError("TG_HOST, TG_GRAPHNAME, and TG_SECRET must be set in .env")

    conn = tg.TigerGraphConnection(
        host=host,
        graphname=graphname,
        gsqlSecret=secret,
    )
    conn.getToken(secret)

    results = conn.runInstalledQuery(
        query_name,
        {"query_vector": embedding},
    )

    print("RAW RESPONSE:")
    print(results)

    print(f"Query vector source: {chunk_id}")
    print("Returned chunk IDs:")
    for result in results:
        rows = result.get("result", []) if isinstance(result, dict) else []
        for row in rows:
            attributes = row.get("attributes", {})
            returned_id = attributes.get("chunk_id", row.get("v_id"))
            if returned_id:
                print(returned_id)


if __name__ == "__main__":
    main()
