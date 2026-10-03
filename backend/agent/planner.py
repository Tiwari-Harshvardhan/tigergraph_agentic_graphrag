AGENT_SYSTEM_PROMPT = """You answer questions about the supplied Olympic corpus.
You can call tools to retrieve structured event facts, temporal relations, graph context,
or relevant text chunks. Choose tools based on the question; you may call multiple tools
and use later calls to resolve IDs or gather supporting evidence. Use event_aggregate for
counts and competitor statistics instead of counting retrieved text yourself. Use only
tool results as factual evidence. When finished, respond with a concise answer and cite
event or chunk IDs where useful. If the corpus does not answer the question, say so.

For event_filter and event_aggregate, map filters to the right fields: sport is the
sport name (such as Biathlon), while discipline is a specific event form (such as
Men's sprint). Use the full edition label when known (such as "2018 Winter Olympics"),
or use year and season separately. For a count with a competitor threshold, set
operation="count" and pass competitor_op and competitor_value; do not put a sport name
in discipline. If an early call returns a plausible set, use a follow-up tool call to
apply any requested numeric condition before answering."""


def _tool(name, description, properties, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


STRING = {"type": "string"}
INTEGER = {"type": "integer", "minimum": 0}
FILTER_PROPERTIES = {
    "sport": {"type": "string", "description": "Sport name, for example Biathlon or Canoeing."},
    "edition": {"type": "string", "description": "Full Olympic edition label, for example 2018 Winter Olympics."},
    "discipline": {"type": "string", "description": "Specific event form, for example Men's sprint; do not use the sport name here."},
    "season": {"type": "string", "enum": ["Summer", "Winter"]},
    "competitor_op": {"type": "string", "enum": [">", ">=", "<", "<=", "="]},
    "competitor_value": INTEGER,
    "year": INTEGER,
}

TOOL_DEFINITIONS = [
    _tool("event_lookup", "Get structured attributes for one Olympic event ID.", {"event_id": STRING}, ["event_id"]),
    _tool("event_filter", "Return events matching structured filters.", FILTER_PROPERTIES),
    _tool(
        "event_aggregate",
        "Aggregate event count or competitor counts for matching events.",
        {"operation": {"type": "string", "enum": ["count", "min", "max", "avg", "sum"]}, **FILTER_PROPERTIES},
        ["operation"],
    ),
    _tool(
        "temporal_search",
        "Return events before, after, between, first, or last by year.",
        {
            "mode": {"type": "string", "enum": ["before", "after", "between", "first", "last"]},
            "from_year": INTEGER,
            "to_year": INTEGER,
            "reference_event_id": STRING,
            "sport": STRING,
            "discipline": STRING,
            "limit": {"type": "integer", "minimum": 1},
        },
        ["mode"],
    ),
    _tool(
        "vector_search",
        "Search corpus chunks semantically; provide the text to embed and search.",
        {"query": STRING, "top_k": {"type": "integer", "minimum": 1}},
        ["query"],
    ),
    _tool(
        "graph_context",
        "Get an event or chunk with its related edition, source document, chunks, and preceding events.",
        {"event_id": STRING, "chunk_id": STRING},
    ),
]
