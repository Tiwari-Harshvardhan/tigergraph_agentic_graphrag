import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

from backend.tigergraph_connection import get_connection


REQUIRED_QUERIES = {
    "event_lookup",
    "event_filter",
    "event_aggregate",
    "temporal_query",
    "vector_search",
    "graph_context",
}
REQUIRED_VERTICES = {"Document", "events", "OlympicEdition", "chunk"}
REQUIRED_EDGES = {"describes", "part_of", "contains", "preceded_by"}


class TigerGraphService:
    def __init__(self, connection_factory=get_connection, sleep=time.sleep):
        load_dotenv()
        self.connection_factory = connection_factory
        self.sleep = sleep
        self.graph_name = os.getenv("TIGERGRAPH_GRAPH") or os.getenv("TG_GRAPHNAME", "wikipedia_corpus")
        self.auto_start = os.getenv("TIGERGRAPH_AUTO_START", "false").lower() == "true"
        self.start_timeout = float(os.getenv("TIGERGRAPH_START_TIMEOUT", "300"))
        self.poll_interval = float(os.getenv("TIGERGRAPH_POLL_INTERVAL", "5"))

    def is_reachable(self) -> bool:
        try:
            return bool(self.connection_factory().echo())
        except Exception:
            return False

    def get_status(self) -> dict:
        status = {
            "reachable": False,
            "graph": self.graph_name,
            "auto_start_enabled": self.auto_start,
            "start_supported": False,
        }
        try:
            connection = self.connection_factory()
            status["reachable"] = bool(connection.echo())
            if status["reachable"]:
                status["installed_queries"] = sorted(
                    key.rsplit("/", 1)[-1] for key in connection.getInstalledQueries()
                )
                status["missing_queries"] = sorted(
                    REQUIRED_QUERIES - set(status["installed_queries"])
                )
        except Exception as error:
            status["error"] = str(error)
        return status

    def start(self) -> dict:
        if self.is_reachable():
            return self.verify()
        raise RuntimeError(
            "TigerGraph workspace start is not available through the configured "
            "TG_HOST/TG_GRAPHNAME/TG_SECRET connection. Start the Savanna workspace "
            "in its control plane, then rerun the command. No start API or control-plane "
            "credentials are configured."
        )

    def wait_until_ready(self, timeout: float | None = None) -> dict:
        timeout = self.start_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        last_error = None
        while True:
            try:
                connection = self.connection_factory()
                if connection.echo():
                    return self.verify(connection=connection)
            except Exception as error:
                last_error = error
            if time.monotonic() >= deadline:
                detail = f" Last error: {last_error}" if last_error else ""
                raise TimeoutError(f"TigerGraph was not ready within {timeout:g} seconds.{detail}")
            self.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))

    def verify(self, connection=None) -> dict:
        connection = connection or self.connection_factory()
        if not connection.echo():
            raise RuntimeError("TigerGraph RESTPP did not respond to echo")

        schema = connection.getSchema()
        graph_name = schema.get("GraphName")
        if graph_name != self.graph_name:
            raise RuntimeError(f"Connected graph is {graph_name!r}; expected {self.graph_name!r}")

        vertices = {vertex["Name"] for vertex in schema.get("VertexTypes", [])}
        edges = {edge["Name"] for edge in schema.get("EdgeTypes", [])}
        missing_vertices = sorted(REQUIRED_VERTICES - vertices)
        missing_edges = sorted(REQUIRED_EDGES - edges)
        if missing_vertices or missing_edges:
            raise RuntimeError(
                f"Graph schema incomplete: missing vertices={missing_vertices}, edges={missing_edges}"
            )

        installed = {
            key.rsplit("/", 1)[-1]: value
            for key, value in connection.getInstalledQueries().items()
        }
        missing_queries = sorted(REQUIRED_QUERIES - installed.keys())
        if missing_queries:
            raise RuntimeError(f"Required TigerGraph queries are not installed: {missing_queries}")

        embedding = next(
            (
                attribute
                for vertex in schema.get("VertexTypes", [])
                if vertex["Name"] == "chunk"
                for attribute in vertex.get("EmbeddingAttributes", [])
                if attribute.get("Name") == "embedding"
            ),
            None,
        )
        if embedding is None or embedding.get("Dimension") != 384:
            raise RuntimeError("The chunk.embedding 384-dimensional vector index is unavailable")

        return {
            "ready": True,
            "graph": graph_name,
            "vertices": sorted(vertices & REQUIRED_VERTICES),
            "edges": sorted(edges & REQUIRED_EDGES),
            "queries": sorted(REQUIRED_QUERIES),
            "embedding_dimension": embedding["Dimension"],
        }

    def ensure_ready(self) -> dict:
        if self.is_reachable():
            return self.verify()
        if self.auto_start:
            self.start()
        raise RuntimeError(
            "TigerGraph is unavailable. Start the wikipedia_corpus Savanna workspace "
            "manually, then rerun. Set TIGERGRAPH_AUTO_START=true only when a supported "
            "Savanna lifecycle API and credentials are configured."
        )


def ensure_tigergraph_ready() -> dict:
    return TigerGraphService().ensure_ready()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check TigerGraph availability and graph readiness.")
    parser.add_argument("command", choices=["status", "check", "start", "wait", "verify"])
    parser.add_argument("--timeout", type=float, default=None)
    arguments = parser.parse_args(argv)
    service = TigerGraphService()

    try:
        if arguments.command == "status":
            result = service.get_status()
        elif arguments.command == "start":
            result = service.start()
        elif arguments.command == "wait":
            result = service.wait_until_ready(arguments.timeout)
        elif arguments.command == "verify":
            result = service.verify()
        else:
            result = service.ensure_ready()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"ready": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())