from backend.tools._common import error_result, event_filter_parameters, run_tool_query


def event_aggregate(
    operation: str,
    *,
    sport=None,
    edition=None,
    discipline=None,
    season=None,
    competitor_op=None,
    competitor_value=None,
    year=None,
):
    operation = operation.lower() if isinstance(operation, str) else ""
    if operation not in {"count", "min", "max", "avg", "sum"}:
        return error_result("event_aggregate", f"Unsupported aggregation operation: {operation!r}")

    try:
        parameters = event_filter_parameters(
            sport=sport,
            edition=edition,
            discipline=discipline,
            season=season,
            competitor_op=competitor_op,
            competitor_value=competitor_value,
            year=year,
        )
    except (TypeError, ValueError) as error:
        return error_result("event_aggregate", error)
    parameters["p_operation"] = operation

    def transform(payload):
        total_count = int(payload.get("total_count", 0))
        metric_key = {
            "count": "total_count",
            "min": "competitor_min",
            "max": "competitor_max",
            "avg": "competitor_avg",
            "sum": "competitor_sum",
        }[operation]
        value = payload.get(metric_key)
        if total_count == 0 and operation in {"min", "max", "avg", "sum"}:
            value = 0 if operation == "sum" else None
        return {
            "operation": operation,
            "value": value,
            "count": total_count,
            "metrics": {
                "competitor_sum": payload.get("competitor_sum", 0),
                "competitor_min": payload.get("competitor_min") if total_count else None,
                "competitor_max": payload.get("competitor_max") if total_count else None,
                "competitor_avg": payload.get("competitor_avg", 0),
            },
        }

    return run_tool_query("event_aggregate", "event_aggregate", parameters, transform)