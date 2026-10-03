from backend.tools._common import error_result, run_tool_query, vertex_rows


def graph_context(*, event_id="", chunk_id=""):
    if not event_id and not chunk_id:
        return error_result("graph_context", "Provide an event_id or chunk_id")
    parameters = {}
    if event_id:
        parameters["p_event_id"] = event_id
    if chunk_id:
        parameters["p_chunk_id"] = chunk_id

    def transform(payload):
        events = vertex_rows(payload, "matched_events")
        return {
            "event": events[0] if events else None,
            "editions": vertex_rows(payload, "editions"),
            "documents": vertex_rows(payload, "event_documents"),
            "chunks": vertex_rows(payload, "event_chunks"),
            "chunk_documents": vertex_rows(payload, "chunk_documents"),
            "selected_chunks": vertex_rows(payload, "selected_chunks"),
            "preceding_events": vertex_rows(payload, "preceding_events"),
        }

    return run_tool_query("graph_context", "graph_context", parameters, transform)