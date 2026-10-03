import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backend.agent.executor import execute_agent
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.services.tigergraph_service import ensure_tigergraph_ready


STATIC_DIR = Path(__file__).resolve().parent / "static"
PIPELINES = {
    "rag": run_rag,
    "graphrag": run_graphrag,
    "agentic": execute_agent,
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
        if self.path == "/health":
            try:
                self._send_json(200, ensure_tigergraph_ready())
            except Exception as error:
                self._send_json(503, {"ready": False, "error": str(error)})
            return
        if self.path not in {"/", "/index.html"}:
            self._send_json(404, {"error": "Not found"})
            return
        body = (STATIC_DIR / "index.html").read_bytes()
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