from backend.tools._common import error_result, event_filter_parameters, run_tool_query, vertex_rows


def event_filter(
    *,
    sport=None,
    edition=None,
    discipline=None,
    season=None,
    competitor_op=None,
    competitor_value=None,
    year=None,
):
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
        return error_result("event_filter", error)

    def transform(payload):
        all_results = vertex_rows(payload, "matched_events")
        return {"results": all_results[:50], "count": len(all_results)}

    return run_tool_query("event_filter", "event_filter", parameters, transform)