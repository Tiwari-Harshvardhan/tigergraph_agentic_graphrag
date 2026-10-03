import json

from backend.agent.planner import AGENT_SYSTEM_PROMPT, TOOL_DEFINITIONS
from backend.agent.state import AgentState
from backend.llm import get_chat_model
from backend.observability import redact_secrets
from backend.pipelines._common import dispatch_tool, failure


_TOOL_NAMES = {
    tool["function"]["name"]
    for tool in TOOL_DEFINITIONS
    if isinstance(tool.get("function"), dict) and tool["function"].get("name")
}


def _contains_tool_call_shaped_text(content: str) -> bool:
    """Detect tool-call-shaped JSON text without executing or rewriting it."""
    decoder = json.JSONDecoder()
    for offset, char in enumerate(content):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[offset:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        name = value.get("name") or value.get("tool")
        arguments = value.get("parameters", value.get("arguments"))
        if name in _TOOL_NAMES and isinstance(arguments, dict):
            return True
    return False


def _failure_with_state(message: str, state: AgentState) -> dict:
    return failure(
        "agentic_graphrag",
        message,
        original_question=state.question,
        evidence=state.evidence,
        tool_calls=state.tool_calls,
        errors=state.tool_errors,
        final_answer=state.final_answer,
        termination_reason=state.termination_reason,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        total_tokens=state.total_tokens,
    )


def execute_agent(
    question: str,
    *,
    model=None,
    max_tool_turns: int = 6,
    max_tool_calls: int = 12,
    max_turns: int | None = None,
) -> dict:
    if max_turns is not None:
        max_tool_turns = max_turns
    if not isinstance(question, str) or not question.strip():
        return failure("agentic_graphrag", "question must be a non-empty string")
    if max_tool_turns < 1 or max_tool_calls < 1:
        return failure("agentic_graphrag", "turn and tool-call limits must be positive")

    state = AgentState(
        question=question,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    try:
        model = model or get_chat_model()
        for _ in range(max_tool_turns):
            try:
                turn = model.complete(state.messages, tools=TOOL_DEFINITIONS)
            except Exception as error:
                state.mark_usage_unavailable()
                from backend.llm import get_model_config

                try:
                    config = get_model_config()
                    provider, model_name = config["provider"], config["model"]
                except Exception:
                    provider, model_name = None, getattr(model, "model", None)
                detail = redact_secrets(str(error))
                state.tool_errors.append({
                    "stage": "model",
                    "request_stage": "after_tool_result" if state.tool_calls else "initial_or_planning",
                    "provider": provider,
                    "model": model_name,
                    "tool_result_being_submitted": bool(state.tool_calls),
                    "termination_reason": "model_error",
                    "error": detail,
                })
                state.termination_reason = "model_error"
                return _failure_with_state(f"Model request failed: {detail}", state)
            state.record_usage(turn)

            if not turn.tool_calls:
                if not turn.content:
                    state.termination_reason = "empty_model_response"
                    return _failure_with_state("The planner returned no answer", state)
                if _contains_tool_call_shaped_text(turn.content):
                    message = "The model emitted tool-call-shaped text instead of a structured tool call"
                    state.termination_reason = "invalid_tool_call_text"
                    state.tool_errors.append({
                        "stage": "model_output",
                        "error_type": "invalid_tool_call_text",
                        "error": message,
                    })
                    return _failure_with_state(message, state)
                state.final_answer = turn.content
                state.termination_reason = "final_answer"
                return {
                    "success": True,
                    "pipeline": "agentic_graphrag",
                    "data": {
                        "answer": turn.content,
                        "original_question": state.question,
                        "final_answer": state.final_answer,
                        "evidence": state.evidence,
                        "tool_calls": state.tool_calls,
                        "prompt_tokens": state.prompt_tokens,
                        "completion_tokens": state.completion_tokens,
                        "total_tokens": state.total_tokens,
                        "errors": state.tool_errors,
                        "termination_reason": state.termination_reason,
                    },
                }

            state.messages.append(
                {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                            "thought_signature": getattr(call, "thought_signature", None),
                        }
                        for call in turn.tool_calls
                    ],
                }
            )
            for call in turn.tool_calls:
                if len(state.tool_calls) >= max_tool_calls:
                    limited_call = {
                        "tool": call.name,
                        "arguments": call.arguments,
                        "result": {"success": False, "error": "Maximum tool-call limit reached"},
                        "success": False,
                        "executed": False,
                    }
                    state.tool_calls.append(limited_call)
                    state.tool_errors.append(
                        {"tool": call.name, "arguments": call.arguments, "error": "Maximum tool-call limit reached"}
                    )
                    state.termination_reason = "max_tool_calls"
                    return _failure_with_state("Maximum tool-call limit reached", state)
                try:
                    result = dispatch_tool(call.name, call.arguments, question)
                except Exception as error:
                    result = {"success": False, "tool": call.name, "error": str(error)}
                call_record = {
                    "tool": call.name,
                    "arguments": call.arguments,
                    "result": result,
                    "success": result.get("success", False),
                }
                state.tool_calls.append(call_record)
                state.evidence.append(result)
                if not result.get("success", False):
                    state.tool_errors.append(
                        {
                            "tool": call.name,
                            "arguments": call.arguments,
                            "error_type": result.get("error_type"),
                            "error": result.get("error"),
                        }
                    )
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        state.termination_reason = "max_tool_turns"
        return _failure_with_state("Maximum tool turns reached", state)
    except Exception as error:
        state.termination_reason = "executor_error"
        state.tool_errors.append({"stage": "executor", "error": str(error)})
        return failure("agentic_graphrag", error)
