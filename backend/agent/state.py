from dataclasses import dataclass, field


@dataclass
class AgentState:
    question: str
    messages: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    tool_errors: list[dict] = field(default_factory=list)
    final_answer: str | None = None
    termination_reason: str | None = None
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    total_tokens: int | None = 0

    def record_usage(self, turn):
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            current = getattr(self, name)
            amount = getattr(turn, name)
            setattr(self, name, current + amount if current is not None and amount is not None else None)

    def mark_usage_unavailable(self):
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None