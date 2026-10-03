import json

def convert_embeddings_to_string(input_path: str, output_path: str):
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)

            emb = chunk.get("embedding")
            if isinstance(emb, list):
                chunk["embedding"] = ",".join(str(float(x)) for x in emb)

            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} records")


if __name__ == "__main__":
    convert_embeddings_to_string(
        "chunks_with_embeddings.json",
        "chunks_with_embeddings_v2.json"
    )