from collections.abc import Callable

from backend.tigergraph_connection import get_connection


def run_installed_query(query_name: str, parameters: dict):
    return get_connection().runInstalledQuery(query_name, parameters)


def run_installed_query_debug(query_name: str, parameters: dict):
    result = get_connection().runInstalledQuery(query_name, parameters)

    print(f"\nQUERY: {query_name}")
    print("=" * 80)
    print(result)
    print("=" * 80)

    return result


def normalize_response(response):
    if isinstance(response, tuple):
        message = ": ".join(str(part) for part in response if part is not None)
        raise RuntimeError(message or "TigerGraph query failed")
    if isinstance(response, dict):
        return response
    if not isinstance(response, list):
        raise RuntimeError(f"Unexpected TigerGraph response: {response!r}")

    payload = {}
    for result in response:
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected TigerGraph result item: {result!r}")
        payload.update(result)
    return payload


def normalize_vertex(vertex: dict) -> dict:
    attributes = vertex.get("attributes")
    if isinstance(attributes, dict):
        normalized = dict(attributes)
    else:
        normalized = {
            key: value
            for key, value in vertex.items()
            if key not in {"v_id", "v_type", "attributes"}
        }

    id_fields = {"event_id", "chunk_id", "doc_id", "edition_id"}
    if not id_fields.intersection(normalized) and vertex.get("v_id") is not None:
        normalized["id"] = vertex["v_id"]
    return normalized


def vertex_rows(payload: dict, result_key: str) -> list[dict]:
    rows = payload.get(result_key, [])
    if rows is None:
        return []
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected a list for {result_key}, got {type(rows).__name__}")
    return [normalize_vertex(row) for row in rows if isinstance(row, dict)]


def success_result(tool: str, data: dict) -> dict:
    return {"success": True, "tool": tool, "data": data}


def error_result(tool: str, error: object) -> dict:
    return {"success": False, "tool": tool, "error": str(error)}


def run_tool_query(
    tool: str,
    query_name: str,
    parameters: dict,
    transform: Callable[[dict], dict],
) -> dict:
    try:
        payload = normalize_response(run_installed_query(query_name, parameters))
        return success_result(tool, transform(payload))
    except Exception as error:
        return error_result(tool, error)


def event_filter_parameters(
    *,
    sport=None,
    edition=None,
    discipline=None,
    season=None,
    competitor_op=None,
    competitor_value=None,
    year=None,
):
    allowed_operators = {"", ">", ">=", "<", "<=", "="}
    operator = competitor_op or ""
    if operator not in allowed_operators:
        raise ValueError(f"Unsupported competitor_op: {operator!r}")
    if operator and competitor_value is None:
        raise ValueError("competitor_value is required when competitor_op is set")
    if not operator and competitor_value is not None:
        raise ValueError("competitor_op is required when competitor_value is set")
    for name, value in (("competitor_value", competitor_value), ("year", year)):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")

    if edition and isinstance(edition, str):
        edition_str = edition.strip().replace("-", " ").title()
        if not edition_str.endswith("Olympics"):
            # Edition labels in the graph include the season (for example,
            # "2018 Winter Olympics"). Preserve it when callers provide the
            # year and season as separate structured filters.
            season_label = season.strip().title() if isinstance(season, str) else ""
            suffix = f" {season_label} Olympics" if season_label else " Olympics"
            edition_str = f"{edition_str}{suffix}"
        edition = edition_str

    return {
        "p_sport": sport or "",
        "p_edition": edition or "",
        "p_discipline": discipline or "",
        "p_season": season or "",
        "p_competitor_op": operator,
        "p_competitor_value": competitor_value or 0,
        "p_year": year or 0,
    }
