__all__ = ["TigerGraphService", "ensure_tigergraph_ready"]


def __getattr__(name):
	if name in __all__:
		from backend.services import tigergraph_service

		return getattr(tigergraph_service, name)
	raise AttributeError(name)