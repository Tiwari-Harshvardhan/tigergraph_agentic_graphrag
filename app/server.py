import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backend.agent.executor import execute_agent
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.services.tigergraph_service import ensure_tigergraph_ready


STATIC_DIR = Path(__file__).resolve().parent / "static"
METRICS_FILE = Path(__file__).resolve().parents[1] / "results" / "benchmark_metrics.json"
PIPELINES = {
    "rag": run_rag,
    "graphrag": run_graphrag,
    "agentic": execute_agent,
}
METRIC_FIELDS = (
    "questions",
    "scored_questions",
    "correct_questions",
    "accuracy",
    "average_latency_seconds",
    "median_latency_seconds",
    "p95_latency_seconds",
    "average_llm_tokens",
    "total_llm_tokens",
    "average_tool_calls",
    "total_tool_calls",
    "failures",
    "failure_rate",
)


def load_benchmark_metrics() -> dict:
    with METRICS_FILE.open(encoding="utf-8") as source:
        report = json.load(source)
    return {
        "model": report["model"],
        "generated_at": report["generated_at"],
        "datasets": [
            {
                "id": dataset["id"],
                "label": dataset["label"],
                "question_count": dataset["question_count"],
                "answer_file": dataset["answer_file"],
                "evaluation_file": dataset["evaluation_file"],
                "pipelines": {
                    name: {field: metrics.get(field) for field in METRIC_FIELDS}
                    for name, metrics in dataset["pipelines"].items()
                },
            }
            for dataset in report["datasets"]
        ],
    }


def run_request(payload: dict) -> dict:
    question = payload.get("question")
    mode = payload.get("mode", "agentic")
    if not isinstance(question, str) or not question.strip():
        return {"success": False, "error": "Enter a question before running."}
    if mode not in {*PIPELINES, "compare"}:
        return {"success": False, "error": f"Unsupported mode: {mode!r}"}

    names = list(PIPELINES) if mode == "compare" else [mode]
    results = {}
    for name in names:
        if name == "rag":
            results[name] = PIPELINES[name](question, top_k=payload.get("top_k", 5))
        elif name == "agentic":
            results[name] = PIPELINES[name](question, max_tool_turns=payload.get("max_tool_turns", 6))
        else:
            results[name] = PIPELINES[name](question)
    return {"success": all(result.get("success", False) for result in results.values()), "results": results}


class DemoHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/metrics":
            try:
                self._send_json(200, load_benchmark_metrics())
            except (OSError, KeyError, json.JSONDecodeError) as error:
                self._send_json(503, {"error": f"Benchmark metrics are unavailable: {error}"})
            return
        if self.path == "/health":
            try:
                self._send_json(200, ensure_tigergraph_ready())
            except Exception as error:
                self._send_json(503, {"ready": False, "error": str(error)})
            return
        pages = {
            "/": "index.html",
            "/index.html": "index.html",
            "/dashboard": "dashboard.html",
            "/dashboard/": "dashboard.html",
        }
        page = pages.get(self.path)
        if page is None:
            self._send_json(404, {"error": "Not found"})
            return
        body = (STATIC_DIR / page).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/run":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 1 or content_length > 100_000:
                self._send_json(413, {"error": "Request body must be between 1 byte and 100 KB."})
                return
            payload = json.loads(self.rfile.read(content_length))
            ensure_tigergraph_ready()
            self._send_json(200, run_request(payload))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Request body must be valid JSON."})
        except Exception as error:
            self._send_json(503, {"error": str(error)})

    def log_message(self, _format, *_args):
        return


def serve(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Olympic GraphRAG console: http://{host}:{server.server_port}")
    server.serve_forever()