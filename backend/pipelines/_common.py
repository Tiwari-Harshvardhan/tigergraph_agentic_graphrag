import json


def sum_token_counts(*counts):
    if any(count is None for count in counts):
        return None
    return sum(counts)


def failure(pipeline: str, error: object, **extra) -> dict:
    result = {"success": False, "pipeline": pipeline, "error": str(error)}
    if extra:
        result["data"] = extra
    return result


def success(pipeline: str, data: dict) -> dict:
    return {"success": True, "pipeline": pipeline, "data": data}


def parse_json_object(content: str) -> dict:
    if not content:
        raise ValueError("The model returned no JSON content")
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from the model")
    return value


def dispatch_tool(name: str, arguments: dict, question: str) -> dict:
    from backend.embeddings import embed_text
    from backend.tools.event_aggregate import event_aggregate
    from backend.tools.event_filter import event_filter
    from backend.tools.event_lookup import event_lookup
    from backend.tools.graph_context import graph_context
    from backend.tools.temporal_search import temporal_search
    from backend.tools.vector_search import vector_search
    from backend.tools.validation import NumericArgumentError, normalize_tool_arguments

    tools = {
        "event_aggregate": event_aggregate,
        "event_filter": event_filter,
        "event_lookup": event_lookup,
        "graph_context": graph_context,
        "temporal_search": temporal_search,
    }
    if name == "vector_search":
        try:
            normalized = normalize_tool_arguments(name, arguments)
        except NumericArgumentError as error:
            return {"success": False, "tool": name, "error_type": "validation_error", "error": str(error)}
        query = normalized.get("query") or question
        return vector_search(embed_text(query), top_k=normalized.get("top_k", 5))
    if name not in tools:
        return {"success": False, "tool": name, "error": "Unknown tool"}

    try:
        args = normalize_tool_arguments(name, arguments)
    except NumericArgumentError as error:
        return {"success": False, "tool": name, "error_type": "validation_error", "error": str(error)}
    # Normalize competitor filters if passed via alternative naming
    if name in {"event_filter", "event_aggregate"}:
        if "competitors_min" in args and "competitor_value" not in args:
            args["competitor_op"] = ">="
            args["competitor_value"] = args.pop("competitors_min")
        elif "competitors_max" in args and "competitor_value" not in args:
            args["competitor_op"] = "<="
            args["competitor_value"] = args.pop("competitors_max")
        if name == "event_aggregate" and "operation" not in args:
            args["operation"] = "count"

    # Only pass keyword arguments supported by the target tool
    param_whitelist = {
        "event_aggregate": {"operation", "sport", "edition", "discipline", "season", "competitor_op", "competitor_value", "year"},
        "event_filter": {"sport", "edition", "discipline", "season", "competitor_op", "competitor_value", "year"},
        "temporal_search": {"mode", "from_year", "to_year", "reference_event_id", "sport", "discipline", "limit"},
        "event_lookup": {"event_id"},
        "graph_context": {"event_id", "chunk_id"},
    }
    allowed = param_whitelist.get(name)
    if allowed:
        args = {k: v for k, v in args.items() if k in allowed}

    return tools[name](**args)


def answer_from_evidence(model, question: str, evidence, system_prompt: str) -> dict:
    response = model.complete(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    "Retrieved evidence (JSON):\n"
                    f"{json.dumps(evidence, ensure_ascii=False, default=str)}"
                ),
            },
        ]
    )
    if response.tool_calls:
        raise RuntimeError("The model attempted a tool call during evidence-only synthesis")
    if not response.content:
        raise RuntimeError("The model returned no answer text")
    prompt_tokens = response.prompt_tokens
    completion_tokens = response.completion_tokens
    return {
        "answer": response.content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": response.total_tokens,
    }
