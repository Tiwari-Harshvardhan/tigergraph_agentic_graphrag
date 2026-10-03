from backend.tools._common import error_result, run_tool_query, vertex_rows


def temporal_search(
    mode: str,
    *,
    from_year=0,
    to_year=0,
    reference_event_id="",
    sport="",
    discipline="",
    limit=10,
):
    mode = mode.lower() if isinstance(mode, str) else ""
    if mode not in {"before", "after", "between", "first", "last"}:
        return error_result("temporal_search", f"Unsupported temporal mode: {mode!r}")
    if mode in {"before", "after"} and not reference_event_id:
        return error_result("temporal_search", "reference_event_id is required for before/after queries")
    if mode == "between" and (not from_year or not to_year or from_year > to_year):
        return error_result("temporal_search", "between queries require from_year <= to_year")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        return error_result("temporal_search", "limit must be a positive integer")

    parameters = {
        "p_mode": mode,
        "p_from_year": from_year,
        "p_to_year": to_year,
        "p_reference_event_id": reference_event_id or "",
        "p_sport": sport,
        "p_discipline": discipline,
        "p_limit": limit,
    }

    def transform(payload):
        results = vertex_rows(payload, "results")
        return {"results": results, "count": len(results)}

    return run_tool_query("temporal_search", "temporal_query", parameters, transform)