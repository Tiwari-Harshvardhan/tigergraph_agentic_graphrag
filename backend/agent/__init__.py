__all__ = ["execute_agent"]


def execute_agent(*args, **kwargs):
	from backend.agent.executor import execute_agent as implementation

	return implementation(*args, **kwargs)