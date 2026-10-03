import time

from backend.llm import get_chat_model
from backend.pipelines._common import (
    answer_from_evidence,
    dispatch_tool,
    failure,
    parse_json_object,
    success,
    sum_token_counts,
)


GRAPH_QUERY_PROMPT = """Translate the user's question into exactly one graph tool call.
Choose among:
- event_aggregate: count or aggregate metrics for events.
  arguments: operation ("count"|"min"|"max"|"avg"|"sum", required), sport (string), edition (string), discipline (string), season ("Summer"|"Winter"), year (integer), competitor_op (">"|">="|"<"|"<="|"="), competitor_value (integer).
- event_filter: list matching events.
  arguments: sport, edition, discipline, season, year, competitor_op, competitor_value.
- event_lookup: look up a specific event by ID.
  arguments: event_id (string).
- temporal_search: search events chronologically.
  arguments: mode ("before"|"after"|"between"|"first"|"last"), reference_event_id (string, required for before/after), from_year (int), to_year (int), sport (string), discipline (string), limit (int).
- graph_context: get related graph neighbors (editions, documents, chunks, preceding events).
  arguments: event_id (string) or chunk_id (string).
- vector_search: semantic text chunk search.
  arguments: query (string), top_k (integer).

Use exact structured filters. For vector_search, provide a concise query string.
Return JSON only: {"tool": "name", "arguments": {}}. Do not answer the user yet."""


ANSWER_PROMPT = (
    "Answer using the supplied graph results as evidence. For deterministic aggregate "
    "results, report the returned value directly. Do not invent facts; cite event IDs "
    "when useful."
)


def run_graphrag(question: str, *, model=None) -> dict:
    started = time.perf_counter()
    if not isinstance(question, str) or not question.strip():
        return failure("graphrag", "question must be a non-empty string")

    planner_started = time.perf_counter()
    planner_attempted = False
    plan_turn = None
    tool_call = None
    evidence_result = None
    synthesis_started = None
    try:
        model = model or get_chat_model()
        planner_started = time.perf_counter()
        planner_attempted = True
        plan_turn = model.complete(
            [
                {"role": "system", "content": GRAPH_QUERY_PROMPT},
                {"role": "user", "content": question},
            ],
            json_mode=True,
        )
        planner_latency = time.perf_counter() - planner_started
        plan = parse_json_object(plan_turn.content)
        tool_name = plan.get("tool")
        arguments = plan.get("arguments", {})
        allowed = {
            "event_lookup",
            "event_filter",
            "event_aggregate",
            "temporal_search",
            "graph_context",
            "vector_search",
        }
        if tool_name not in allowed or not isinstance(arguments, dict):
            raise ValueError("The planner returned an unsupported tool or arguments")

        tool_call = {"tool": tool_name, "arguments": arguments}
        evidence_result = dispatch_tool(tool_name, arguments, question)
        if not evidence_result["success"]:
            return failure(
                "graphrag",
                evidence_result["error"],
                answer=None,
                evidence={"tool": tool_name, "data": evidence_result},
                tool_calls=[tool_call],
                prompt_tokens=plan_turn.prompt_tokens,
                completion_tokens=plan_turn.completion_tokens,
                total_tokens=plan_turn.total_tokens,
                termination_reason=(
                    "tool_validation_error"
                    if evidence_result.get("error_type") == "validation_error"
                    else "tool_error"
                ),
                planner_latency_seconds=planner_latency,
                llm_latency_seconds=0,
                latency_seconds=time.perf_counter() - started,
            )

        evidence = {"tool": tool_name, "data": evidence_result["data"]}
        synthesis_started = time.perf_counter()
        # Keep orchestration metadata out of the synthesis context. In
        # particular, the "tool" label describes a completed operation; it is
        # not an instruction to call that operation again.
        answer = answer_from_evidence(model, question, evidence_result["data"], ANSWER_PROMPT)
        synthesis_latency = time.perf_counter() - synthesis_started
        prompt_tokens = sum_token_counts(plan_turn.prompt_tokens, answer["prompt_tokens"])
        completion_tokens = sum_token_counts(plan_turn.completion_tokens, answer["completion_tokens"])
        total_tokens = sum_token_counts(plan_turn.total_tokens, answer["total_tokens"])
        return success(
            "graphrag",
            {
                **answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "evidence": evidence,
                "tool_calls": [tool_call],
                "planner_usage": {
                    "prompt_tokens": plan_turn.prompt_tokens,
                    "completion_tokens": plan_turn.completion_tokens,
                    "total_tokens": plan_turn.total_tokens,
                },
                "planner_latency_seconds": planner_latency,
                "llm_latency_seconds": synthesis_latency,
                "latency_seconds": time.perf_counter() - started,
            },
        )
    except Exception as error:
        synthesis_failed = synthesis_started is not None
        planner_prompt = plan_turn.prompt_tokens if plan_turn else (None if planner_attempted else 0)
        planner_completion = plan_turn.completion_tokens if plan_turn else (None if planner_attempted else 0)
        planner_total = plan_turn.total_tokens if plan_turn else (None if planner_attempted else 0)
        return failure(
            "graphrag",
            error,
            answer=None,
            evidence={"tool": tool_call, "result": evidence_result} if evidence_result else None,
            tool_calls=[tool_call] if tool_call else [],
            prompt_tokens=None if synthesis_failed else planner_prompt,
            completion_tokens=None if synthesis_failed else planner_completion,
            total_tokens=None if synthesis_failed else planner_total,
            termination_reason="model_error" if synthesis_failed else "planner_error",
            planner_latency_seconds=time.perf_counter() - planner_started if planner_attempted else 0,
            llm_latency_seconds=time.perf_counter() - synthesis_started if synthesis_failed else 0,
            latency_seconds=time.perf_counter() - started,
        )
