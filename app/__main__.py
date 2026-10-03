import argparse
import json
import sys
from datetime import datetime, timezone

from backend.agent.executor import execute_agent
from backend.observability import configure_logging, log_run, new_run_id
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.services.tigergraph_service import ensure_tigergraph_ready


PIPELINES = {
    "rag": run_rag,
    "graphrag": run_graphrag,
    "agentic": execute_agent,
}


def _run_pipeline(pipeline_name, question):
    if pipeline_name == "agentic":
        return PIPELINES[pipeline_name](question, max_tool_turns=6)
    return PIPELINES[pipeline_name](question)


def main(argv=None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Readiness-gated Olympic GraphRAG console demo")
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="agentic")
    parser.add_argument("--check", action="store_true", help="Verify dependencies and exit")
    parser.add_argument("--web", action="store_true", help="Start the local browser demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.web:
        from app.server import serve

        serve(args.host, args.port)
        return 0

    try:
        readiness = ensure_tigergraph_ready()
    except Exception as error:
        print(json.dumps({"ready": False, "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    if args.check or not sys.stdin.isatty():
        return 0

    print(f"Ready. Pipeline: {args.pipeline}. Enter a question; submit an empty line to exit.")
    while True:
        try:
            question = input("Question> ").strip()
        except EOFError:
            break
        if not question:
            break
        result = _run_pipeline(args.pipeline, question)
        log_run(
            {
                "run_id": new_run_id(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline": args.pipeline,
                "question": question,
                "result": result,
            }
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())