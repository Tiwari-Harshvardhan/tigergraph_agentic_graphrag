import json
from sentence_transformers import SentenceTransformer

def build_chunk_embeddings(chunks_path: str, output_path: str, model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)  # 384-dim, runs locally, no API key needed

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for c, emb in zip(chunks, embeddings):
            c["embedding"] = emb.tolist()
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Written {len(chunks)} chunks with embeddings, dim={embeddings.shape[1]}")


if __name__ == "__main__":
    build_chunk_embeddings("chunks.json", "chunks_with_embeddings.json")