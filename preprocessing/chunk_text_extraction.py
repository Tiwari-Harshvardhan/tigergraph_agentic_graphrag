import json

def chunk_text(text: str, chunk_size_words: int = 200, overlap_words: int = 40):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def build_chunks(documents_path: str, output_path: str):
    total_chunks = 0
    with open(documents_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue

            pieces = chunk_text(text)
            for idx, piece in enumerate(pieces):
                chunk = {
                    "chunk_id": f"{doc['doc_id']}_c{idx}",
                    "document_id": doc["doc_id"],
                    "chunk_index": idx,
                    "text": piece,
                    "token_count": len(piece.split()),  # word-count proxy; swap for a real tokenizer if you have one
                }
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Chunks written: {total_chunks}")


if __name__ == "__main__":
    build_chunks("documents.json", "chunks.json")