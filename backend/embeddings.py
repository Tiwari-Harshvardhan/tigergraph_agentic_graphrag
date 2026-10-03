import os


_models = {}


def embed_text(text: str) -> list[float]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    model = _models.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _models[model_name] = model

    vector = model.encode(text, convert_to_numpy=True)
    if len(vector) != 384:
        raise ValueError(f"Expected a 384-dimensional embedding, got {len(vector)}")
    return [float(value) for value in vector]