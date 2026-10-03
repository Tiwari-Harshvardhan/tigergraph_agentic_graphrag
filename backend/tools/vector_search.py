import math

from backend.tools._common import error_result, run_tool_query, vertex_rows


def vector_search(query_vector, top_k=5):
    try:
        vector = [float(value) for value in query_vector]
    except (TypeError, ValueError) as error:
        return error_result("vector_search", f"query_vector must contain numeric values: {error}")
    if len(vector) != 384:
        return error_result(
            "vector_search",
            f"query_vector must contain 384 values, got {len(vector)}",
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        return error_result("vector_search", "top_k must be a positive integer")

    if not all(math.isfinite(value) for value in vector):
        return error_result("vector_search", "query_vector values must be finite")

    def transform(payload):
        results = vertex_rows(payload, "results")
        return {"results": results, "count": len(results)}

    return run_tool_query(
        "vector_search",
        "vector_search",
        {"query_vector": vector, "top_k": top_k},
        transform,
    )