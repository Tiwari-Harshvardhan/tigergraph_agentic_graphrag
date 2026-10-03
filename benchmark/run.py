import argparse
import json
import multiprocessing
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.pipelines.agentic_graphrag import run_agentic_graphrag
from backend.pipelines.graphrag import run_graphrag
from backend.pipelines.rag import run_rag
from backend.observability import append_jsonl, configure_logging, log_run, redact_secrets
from backend.llm import get_model_config
from backend.services.tigergraph_service import ensure_tigergraph_ready
from evaluation.answers import evaluate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    PROJECT_ROOT / "questions" / "eval_public.jsonl",
    PROJECT_ROOT / "questions" / "eval_hidden.jsonl",
]
PIPELINES = {
    "rag": run_rag,
    "graphrag": run_graphrag,
    "agentic": run_agentic_graphrag,
}


def load_questions(paths, *, limit=None, question_ids=None):
    questions = []
    selected_ids = set(question_ids or [])
    for path in paths:
        input_path = Path(path)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        with input_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                question_id = record.get("qid", record.get("id"))
                question = record.get("question")
                if not question_id or not question:
                    raise ValueError(f"{input_path}:{line_number} requires qid/id and question")
                if selected_ids and question_id not in selected_ids:
                    continue
                questions.append({**record, "_question_id": question_id})

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    if not questions:
        raise ValueError("No benchmark questions matched the selected input and filters")
    return questions


def _pipeline_call(name, question, *, top_k, max_tool_turns):
    if name == "rag":
        return PIPELINES[name](question, top_k=top_k)
    if name == "agentic":
        return PIPELINES[name](question, max_tool_turns=max_tool_turns)
    return PIPELINES[name](question)


def _run_one(task):
    record, pipeline_name, run_id, timestamp, top_k, max_tool_turns = task
    started = time.perf_counter()
    thrown_error = None
    try:
        result = _pipeline_call(
            pipeline_name,
            record["question"],
            top_k=top_k,
            max_tool_turns=max_tool_turns,
        )
    except Exception as error:
        thrown_error = error
        result = {"success": False, "error": str(error), "data": {}}
    elapsed = time.perf_counter() - started
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    tool_calls = data.get("tool_calls") or []
    expected_answer = record.get("answer")
    answer_evaluation = (
        evaluate_answer(
            data.get("answer"),
            expected_answer,
            record.get("qtype", record.get("type")),
        )
        if result.get("success")
        else {
            "correct": None,
            "method": "operational_failure",
            "normalized_generated": "",
        }
    )
    prompt_tokens = data.get("prompt_tokens")
    completion_tokens = data.get("completion_tokens")
    total_tokens = data.get("total_tokens")
    model_config = get_model_config()
    error_text = result.get("error") if not result.get("success") else None
    if not error_text and not result.get("success") and data.get("errors"):
        last_error = data["errors"][-1]
        error_text = last_error.get("error") if isinstance(last_error, dict) else str(last_error)
    diagnostic = {}
    if error_text:
        for key, pattern in (
            ("provider", r"provider=([\w.-]+)"),
            ("model", r"model=([^ ]+)"),
            ("error_type", r"error_type=([\w.-]+)"),
            ("http_status", r"status=(\d{3})"),
            ("request_type", r"request_type=([\w.-]+)"),
        ):
            match = re.search(pattern, str(error_text))
            diagnostic[key] = match.group(1) if match else None
        if thrown_error is not None:
            diagnostic["exception_type"] = type(thrown_error).__name__
        diagnostic["termination_reason"] = data.get("termination_reason")
    safe_error = redact_secrets(error_text) if error_text else None
    safe_diagnostic = redact_secrets(diagnostic) if error_text else None
    error_status = safe_diagnostic.get("http_status") if safe_diagnostic else None
    failure_category = "provider_credit" if error_status == "402" else None
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "question_id": record["_question_id"],
        "question": record["question"],
        "expected_answer": expected_answer,
        "pipeline": pipeline_name,
        "provider": model_config["provider"],
        "model": model_config["model"],
        "generated_answer": data.get("answer"),
        "correctness": answer_evaluation["correct"],
        "evaluation": answer_evaluation,
        "latency_seconds": data.get("latency_seconds", elapsed),
        "llm_latency_seconds": data.get("llm_latency_seconds"),
        "retrieval_latency_seconds": data.get("retrieval_latency_seconds"),
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "available": any(value is not None for value in (prompt_tokens, completion_tokens, total_tokens)),
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "evidence": data.get("evidence"),
        "termination_reason": data.get("termination_reason"),
        "error": safe_error,
        "error_type": safe_diagnostic.get("error_type") or safe_diagnostic.get("exception_type") if safe_diagnostic else None,
        "error_status": safe_diagnostic.get("http_status") if safe_diagnostic else None,
        "failure_category": failure_category,
        "error_message": safe_error,
        "diagnostic": safe_diagnostic,
        "status": "success" if result.get("success") else "failed",
        "success": bool(result.get("success")),
        "question_type": record.get("qtype", record.get("type")),
        "metadata": {
            "llm_provider": model_config["provider"],
            "llm_model": model_config["model"],
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "top_k": top_k if pipeline_name == "rag" else None,
            "max_tool_turns": max_tool_turns if pipeline_name == "agentic" else None,
            "temperature": model_config["temperature"],
            "graph": os.getenv("TIGERGRAPH_GRAPH") or os.getenv("TG_GRAPHNAME", "wikipedia_corpus"),
        },
    }


def _run_one_worker(task, connection):
    try:
        connection.send(_run_one(task))
    finally:
        connection.close()


def _timeout_result(task, timeout_seconds):
    record, pipeline_name, run_id, timestamp, top_k, max_tool_turns = task
    model_config = get_model_config()
    error = f"Pipeline execution exceeded the {timeout_seconds:g}-second benchmark timeout"
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "question_id": record["_question_id"],
        "question": record["question"],
        "expected_answer": record.get("answer"),
        "pipeline": pipeline_name,
        "provider": model_config["provider"],
        "model": model_config["model"],
        "generated_answer": None,
        "correctness": None,
        "evaluation": {
            "correct": None,
            "method": "operational_failure",
            "normalized_generated": "",
        },
        "latency_seconds": timeout_seconds,
        "llm_latency_seconds": None,
        "retrieval_latency_seconds": None,
        "token_usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "available": False,
        },
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "tool_call_count": 0,
        "tool_calls": [],
        "evidence": None,
        "termination_reason": "execution_timeout",
        "error": error,
        "error_type": "ExecutionTimeout",
        "error_status": None,
        "failure_category": "execution_timeout",
        "error_message": error,
        "diagnostic": {
            "exception_type": "ExecutionTimeout",
            "termination_reason": "execution_timeout",
            "timeout_seconds": timeout_seconds,
        },
        "status": "failed",
        "success": False,
        "question_type": record.get("qtype", record.get("type")),
        "metadata": {
            "llm_provider": model_config["provider"],
            "llm_model": model_config["model"],
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "top_k": top_k if pipeline_name == "rag" else None,
            "max_tool_turns": max_tool_turns if pipeline_name == "agentic" else None,
            "temperature": model_config["temperature"],
            "graph": os.getenv("TIGERGRAPH_GRAPH") or os.getenv("TG_GRAPHNAME", "wikipedia_corpus"),
        },
    }


def _worker_exit_result(task):
    record, pipeline_name, run_id, timestamp, top_k, max_tool_turns = task
    model_config = get_model_config()
    error = "Pipeline worker exited without returning a result"
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "question_id": record["_question_id"],
        "question": record["question"],
        "expected_answer": record.get("answer"),
        "pipeline": pipeline_name,
        "provider": model_config["provider"],
        "model": model_config["model"],
        "generated_answer": None,
        "correctness": None,
        "evaluation": {"correct": None, "method": "operational_failure", "normalized_generated": ""},
        "latency_seconds": None,
        "llm_latency_seconds": None,
        "retrieval_latency_seconds": None,
        "token_usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "available": False},
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "tool_call_count": 0,
        "tool_calls": [],
        "evidence": None,
        "termination_reason": "worker_exit",
        "error": error,
        "error_type": "WorkerExit",
        "error_status": None,
        "failure_category": "worker_exit",
        "error_message": error,
        "diagnostic": {"exception_type": "WorkerExit", "termination_reason": "worker_exit"},
        "status": "failed",
        "success": False,
        "question_type": record.get("qtype", record.get("type")),
        "metadata": {
            "llm_provider": model_config["provider"],
            "llm_model": model_config["model"],
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "top_k": top_k if pipeline_name == "rag" else None,
            "max_tool_turns": max_tool_turns if pipeline_name == "agentic" else None,
            "temperature": model_config["temperature"],
            "graph": os.getenv("TIGERGRAPH_GRAPH") or os.getenv("TG_GRAPHNAME", "wikipedia_corpus"),
        },
    }


def _run_one_with_timeout(task, timeout_seconds):
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    worker = context.Process(target=_run_one_worker, args=(task, send_connection))
    started = time.perf_counter()
    worker.start()
    send_connection.close()
    try:
        if not receive_connection.poll(timeout_seconds):
            worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()
            result = _timeout_result(task, timeout_seconds)
            result["latency_seconds"] = time.perf_counter() - started
            return result
        try:
            result = receive_connection.recv()
        except EOFError:
            result = _worker_exit_result(task)
        worker.join()
        return result
    finally:
        receive_connection.close()
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)


def _read_completed(output_path):
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open(encoding="utf-8") as existing:
        for line in existing:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("success") is True:
                completed.add((row.get("pipeline"), row.get("question_id")))
    return completed


def run_benchmark(
    *,
    pipelines,
    input_paths=None,
    output_path=None,
    limit=None,
    question_ids=None,
    resume=False,
    parallel=False,
    workers=2,
    top_k=5,
    max_tool_turns=6,
    execution_timeout_seconds=None,
    ensure_ready=True,
):
    unknown = set(pipelines) - set(PIPELINES)
    if unknown:
        raise ValueError(f"Unsupported pipelines: {sorted(unknown)}")
    if parallel:
        raise ValueError("Parallel runs cannot guarantee an immediate stop on provider credit errors; use sequential mode")
    if execution_timeout_seconds is not None and execution_timeout_seconds <= 0:
        raise ValueError("execution_timeout_seconds must be positive")
    if ensure_ready:
        ensure_tigergraph_ready()

    questions = load_questions(input_paths or DEFAULT_INPUTS, limit=limit, question_ids=question_ids)
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "runs" / f"{run_id}.jsonl"
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    source_paths = [Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path for path in (input_paths or DEFAULT_INPUTS)]
    if output_path.resolve() in {path.resolve() for path in source_paths}:
        raise ValueError("Output path must not overwrite an input benchmark file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not resume:
        raise FileExistsError(f"Result file already exists; use --resume to append safely: {output_path}")

    completed = _read_completed(output_path) if resume else set()
    tasks = [
        (question, pipeline, run_id, timestamp, top_k, max_tool_turns)
        for question in questions
        for pipeline in pipelines
        if (pipeline, question["_question_id"]) not in completed
    ]
    if not output_path.exists():
        output_path.touch()
    executed = 0
    for task in tasks:
        result = (
            _run_one(task)
            if execution_timeout_seconds is None
            else _run_one_with_timeout(task, execution_timeout_seconds)
        )
        log_run(_log_record(result))
        append_jsonl(output_path, result)
        executed += 1
        if result.get("error_status") == "402":
            break
        time.sleep(1.0)
    return {"run_id": run_id, "output": str(output_path), "question_count": len(questions), "result_count": executed}


def _log_record(result):
    fields = (
        "run_id",
        "timestamp",
        "question_id",
        "pipeline",
        "success",
        "latency_seconds",
        "token_usage",
        "tool_call_count",
        "tool_calls",
        "evidence",
        "error",
        "metadata",
    )
    return {key: result.get(key) for key in fields}


def main(argv=None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run RAG, GraphRAG, and Agentic GraphRAG benchmarks.")
    parser.add_argument("--pipeline", choices=["rag", "graphrag", "agentic", "all"], default="all")
    parser.add_argument("--input", action="append", dest="inputs", help="JSONL file; repeat to combine inputs")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--parallel", action="store_true")
    mode.add_argument("--sequential", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tool-turns", type=int, default=6)
    parser.add_argument("--execution-timeout", type=float, default=None, help="Maximum seconds allowed for each pipeline/question execution")
    args = parser.parse_args(argv)
    pipelines = list(PIPELINES) if args.pipeline == "all" else [args.pipeline]
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


if __name__ == "__main__":
    sys.exit(main())
