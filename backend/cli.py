import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from backend.agent.executor import execute_agent
from backend.llm import get_model_config
from backend.observability import append_jsonl, configure_logging, log_run, new_run_id
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.services.tigergraph_service import TigerGraphService


def _service_command(args) -> int:
	service = TigerGraphService()
	try:
		if args.action == "status":
			result = service.get_status()
		elif args.action == "start":
			result = service.start()
		elif args.action == "wait":
			result = service.wait_until_ready(args.timeout)
		elif args.action == "verify":
			result = service.verify()
		else:
			result = service.ensure_ready()
		print(json.dumps(result, ensure_ascii=False, indent=2))
		return 0
	except Exception as error:
		print(json.dumps({"success": False, "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
		return 1


def _run_command(args) -> int:
	try:
		TigerGraphService().ensure_ready()
		if args.pipeline == "rag":
			result = run_rag(args.question, top_k=args.top_k)
		elif args.pipeline == "graphrag":
			result = run_graphrag(args.question)
		else:
			result = execute_agent(args.question, max_tool_turns=args.max_tool_turns)

		run_id = new_run_id()
		model_config = get_model_config()
		record = {
			"run_id": run_id,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"question_id": args.question_id,
			"question": args.question,
			"pipeline": args.pipeline,
			"success": result.get("success", False),
			"generated_answer": result.get("data", {}).get("answer"),
			"error": result.get("error"),
			"data": result.get("data"),
			"graph": os.getenv("TIGERGRAPH_GRAPH") or os.getenv("TG_GRAPHNAME", "wikipedia_corpus"),
			"model": model_config,
		}
		log_run(record)
		if args.output:
			append_jsonl(args.output, record)
		print(json.dumps(record, ensure_ascii=False, indent=2, default=str))
		return 0 if result.get("success") else 1
	except Exception as error:
		print(json.dumps({"success": False, "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
		return 1


def main(argv=None) -> int:
	load_dotenv()
	configure_logging()
	parser = argparse.ArgumentParser(description="Olympic TigerGraph RAG comparison tools")
	commands = parser.add_subparsers(dest="command", required=True)

	tigergraph = commands.add_parser("tigergraph", help="TigerGraph readiness operations")
	tg_actions = tigergraph.add_subparsers(dest="action", required=True)
	for action in ("status", "check", "start", "wait", "verify"):
		action_parser = tg_actions.add_parser(action)
		if action == "wait":
			action_parser.add_argument("--timeout", type=float, default=None)
	tigergraph.set_defaults(handler=_service_command)

	run = commands.add_parser("run", help="Run one pipeline for a question")
	run.add_argument("pipeline", choices=["rag", "graphrag", "agentic"])
	run.add_argument("question")
	run.add_argument("--question-id")
	run.add_argument("--output")
	run.add_argument("--top-k", type=int, default=5)
	run.add_argument("--max-tool-turns", type=int, default=6)
	run.set_defaults(handler=_run_command)

	benchmark = commands.add_parser("benchmark", help="Run the benchmark")
	benchmark.add_argument("--pipeline", choices=["rag", "graphrag", "agentic", "all"], default="all")
	benchmark.add_argument("--input", action="append", dest="inputs")
	benchmark.add_argument("--output")
	benchmark.add_argument("--limit", type=int)
	benchmark.add_argument("--question-id", action="append", dest="question_ids")
	benchmark.add_argument("--resume", action="store_true")
	benchmark_mode = benchmark.add_mutually_exclusive_group()
	benchmark_mode.add_argument("--parallel", action="store_true")
	benchmark_mode.add_argument("--sequential", action="store_true")
	benchmark.add_argument("--workers", type=int, default=2)
	benchmark.add_argument("--top-k", type=int, default=5)
	benchmark.add_argument("--max-tool-turns", type=int, default=6)
	benchmark.add_argument("--execution-timeout", type=float, default=None, help="Maximum seconds allowed for each pipeline/question execution")
	benchmark.set_defaults(handler=_benchmark_command)

	evaluate = commands.add_parser("evaluate", help="Evaluate saved result JSONL")
	evaluate.add_argument("--input", required=True)
	evaluate.add_argument("--output")
	evaluate.set_defaults(handler=_evaluate_command)

	test = commands.add_parser("test", help="Run the offline test suite")
	test.set_defaults(handler=_test_command)

	args = parser.parse_args(argv)
	return args.handler(args)


def _benchmark_command(args) -> int:
	from benchmark.run import run_benchmark

	pipelines = ["rag", "graphrag", "agentic"] if args.pipeline == "all" else [args.pipeline]
	try:
		result = run_benchmark(
			pipelines=pipelines,
			input_paths=args.inputs,
			output_path=args.output,
			limit=args.limit,
			question_ids=args.question_ids,
			resume=args.resume,
			parallel=args.parallel,
			workers=args.workers,
			top_k=args.top_k,
			max_tool_turns=args.max_tool_turns,
			execution_timeout_seconds=args.execution_timeout,
		)
		print(json.dumps(result, ensure_ascii=False, indent=2))
		return 0
	except Exception as error:
		print(json.dumps({"success": False, "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
		return 1


def _evaluate_command(args) -> int:
	from evaluation.evaluate import main as evaluate_main

	forwarded = ["--input", args.input]
	if args.output:
		forwarded.extend(["--output", args.output])
	return evaluate_main(forwarded)


def _test_command(_args) -> int:
	root = Path(__file__).resolve().parents[1]
	return subprocess.run(
		[sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
		cwd=root,
		check=False,
	).returncode


if __name__ == "__main__":
	sys.exit(main())
