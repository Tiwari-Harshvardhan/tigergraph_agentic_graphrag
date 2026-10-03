__all__ = ["run_agentic_graphrag", "run_graphrag", "run_rag"]


def run_rag(*args, **kwargs):
	from backend.pipelines.rag import run_rag as implementation

	return implementation(*args, **kwargs)


def run_graphrag(*args, **kwargs):
	from backend.pipelines.graphrag import run_graphrag as implementation

	return implementation(*args, **kwargs)


def run_agentic_graphrag(*args, **kwargs):
	from backend.pipelines.agentic_graphrag import run_agentic_graphrag as implementation

	return implementation(*args, **kwargs)