import time

from backend.embeddings import embed_text
from backend.llm import get_chat_model
from backend.pipelines._common import answer_from_evidence, failure, success
from backend.tools.vector_search import vector_search


RAG_SYSTEM_PROMPT = (
    "Answer the user's question using only the retrieved corpus chunks. "
    "If the evidence is insufficient, say so. Cite supporting chunk IDs in square brackets."
)


def run_rag(question: str, *, model=None, embedder=embed_text, top_k: int = 5) -> dict:
    started = time.perf_counter()
    if not isinstance(question, str) or not question.strip():
        return failure("rag", "question must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        return failure("rag", "top_k must be a positive integer")

    embedding_latency = 0
    retrieval_latency = 0
    synthesis_started = None
    evidence = []
    tool_calls = []
    try:
        model = model or get_chat_model()
        embedding_started = time.perf_counter()
        query_vector = embedder(question)
        embedding_latency = time.perf_counter() - embedding_started
        retrieval_started = time.perf_counter()
        retrieval = vector_search(query_vector, top_k=top_k)
        retrieval_latency = time.perf_counter() - retrieval_started
        tool_calls.append(
            {
                "tool": "vector_search",
                "success": retrieval["success"],
                "count": retrieval.get("data", {}).get("count", 0),
                "error": retrieval.get("error"),
            }
        )
        if not retrieval["success"]:
            return failure(
                "rag",
                retrieval["error"],
                answer=None,
                evidence=[],
                tool_calls=tool_calls,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                embedding_latency_seconds=embedding_latency,
                retrieval_latency_seconds=retrieval_latency,
                llm_latency_seconds=None,
                latency_seconds=time.perf_counter() - started,
            )

        evidence = retrieval["data"]["results"]
        synthesis_started = time.perf_counter()
        answer = answer_from_evidence(model, question, evidence, RAG_SYSTEM_PROMPT)
        synthesis_latency = time.perf_counter() - synthesis_started
        return success(
            "rag",
            {
                **answer,
                "evidence": evidence,
                "tool_calls": [{"tool": "vector_search", "count": len(evidence)}],
                "embedding_latency_seconds": embedding_latency,
                "retrieval_latency_seconds": retrieval_latency,
                "llm_latency_seconds": synthesis_latency,
                "latency_seconds": time.perf_counter() - started,
            },
        )
    except Exception as error:
        llm_attempted = synthesis_started is not None
        return failure(
            "rag",
            error,
            answer=None,
            evidence=evidence,
            tool_calls=tool_calls,
            prompt_tokens=None if llm_attempted else 0,
            completion_tokens=None if llm_attempted else 0,
            total_tokens=None if llm_attempted else 0,
            embedding_latency_seconds=embedding_latency,
            retrieval_latency_seconds=retrieval_latency,
            llm_latency_seconds=time.perf_counter() - synthesis_started if llm_attempted else 0,
            latency_seconds=time.perf_counter() - started,
        )