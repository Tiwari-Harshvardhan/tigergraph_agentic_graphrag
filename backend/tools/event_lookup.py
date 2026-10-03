from backend.tools._common import error_result, run_tool_query, vertex_rows


def event_lookup(event_id: str):
    if not event_id:
        return error_result("event_lookup", "event_id is required")

    def transform(payload):
        events = vertex_rows(payload, "matched_events")
        return {
            "event": events[0] if events else None,
            "editions": vertex_rows(payload, "editions"),
            "found": bool(events),
        }

    return run_tool_query("event_lookup", "event_lookup", {"event_id": event_id}, transform)