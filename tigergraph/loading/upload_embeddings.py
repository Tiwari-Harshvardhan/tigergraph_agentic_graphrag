import os
from pathlib import Path
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("TG_HOST")
GRAPHNAME = os.getenv("TG_GRAPHNAME")
SECRET = os.getenv("TG_SECRET")

print("Connecting to TigerGraph...")

conn = tg.TigerGraphConnection(
    host=HOST,
    graphname=GRAPHNAME,
    gsqlSecret=SECRET
)

conn.getToken(SECRET)

print("Connection successful.")
print(conn.echo())

print("\nUploading embeddings...")

loading_job = "load_chunk_embeddings"
if loading_job not in conn.getLoadingJobs():
    raise RuntimeError(
        f"Loading job '{loading_job}' is not defined for graph '{GRAPHNAME}'. "
        "Create and enable a job that maps chunk_id and embedding to vertex chunk."
    )

input_path = Path("corpus/chunks_with_embeddings_v2.json")
batch_limit = 1 * 1024 * 1024
embedding_dimension = 384
batch_lines = []
batch_size = 0
batch_number = 0
record_count = 0
chunk_ids = set()
results = []

with input_path.open(encoding="utf-8") as input_file:
    header = input_file.readline().rstrip("\r\n")
    if header != "chunk_id|embedding":
        raise ValueError(f"Unexpected input header: {header!r}")

    for line_number, line in enumerate(input_file, start=2):
        line = line.rstrip("\r\n")
        fields = line.split("|", 1)
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(f"Invalid pipe-delimited record at line {line_number}")

        chunk_id, embedding_text = fields
        if chunk_id in chunk_ids:
            raise ValueError(f"Duplicate chunk_id at line {line_number}: {chunk_id}")
        chunk_ids.add(chunk_id)

        embedding_values = embedding_text.split(",")
        if len(embedding_values) != embedding_dimension:
            raise ValueError(
                f"Expected {embedding_dimension} values at line {line_number}, "
                f"found {len(embedding_values)}"
            )
        try:
            [float(value) for value in embedding_values]
        except ValueError as error:
            raise ValueError(f"Non-numeric embedding at line {line_number}") from error

        record_count += 1
        line += "\n"
        encoded_line = line.encode("utf-8")
        if batch_lines and batch_size + len(encoded_line) > batch_limit:
            batch_number += 1
            print(f"Uploading batch {batch_number}...")
            result = conn.runLoadingJobWithData(
                "".join(batch_lines),
                "chunk_file",
                loading_job,
                "|",
                timeout=600000,
                sizeLimit=batch_limit,
            )
            results.append(result)
            print(result)
            batch_lines = []
            batch_size = 0
        batch_lines.append(line)
        batch_size += len(encoded_line)

if batch_lines:
    batch_number += 1
    print(f"Uploading batch {batch_number}...")
    result = conn.runLoadingJobWithData(
        "".join(batch_lines),
        "chunk_file",
        loading_job,
        "|",
        timeout=600000,
        sizeLimit=batch_limit,
    )
    results.append(result)
    print(result)

print("\nLoading result:")
print(results)
print(f"Uploaded {record_count} validated records in {batch_number} batches.")