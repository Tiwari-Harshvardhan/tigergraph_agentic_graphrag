import time

from backend.agent.executor import execute_agent


def run_agentic_graphrag(
    question: str,
    *,
    model=None,
    max_tool_turns: int = 6,
    max_turns: int | None = None,
) -> dict:
    started = time.perf_counter()
    result = execute_agent(
        question,
        model=model,
        max_tool_turns=max_tool_turns,
        max_turns=max_turns,
    )
    if not isinstance(result.get("data"), dict):
        result["data"] = {
            "answer": None,
            "evidence": [],
            "tool_calls": [],
            "errors": [result.get("error")],
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    result["data"]["latency_seconds"] = time.perf_counter() - started
    return result