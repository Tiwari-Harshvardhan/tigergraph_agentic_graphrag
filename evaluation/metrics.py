import math
import statistics
from collections import defaultdict

from evaluation.answers import evaluate_answer


def _number(values, key):
    return [value[key] for value in values if isinstance(value.get(key), (int, float))]


def _summary(rows):
    count = len(rows)
    scored = [row for row in rows if isinstance(row.get("correctness"), bool)]
    correct = sum(row["correctness"] is True for row in scored)
    latencies = _number(rows, "latency_seconds")
    llm_token_values = [
        row.get("token_usage", {}).get("total_tokens")
        for row in rows
        if isinstance(row.get("token_usage"), dict)
        and isinstance(row["token_usage"].get("total_tokens"), int)
    ]
    tool_calls = _number(rows, "tool_call_count")
    failures = sum(not row.get("success", True) or bool(row.get("error")) for row in rows)
    latency_p95 = None
    if len(latencies) >= 20:
        sorted_latencies = sorted(latencies)
        latency_p95 = sorted_latencies[math.ceil(0.95 * len(sorted_latencies)) - 1]
    return {
        "questions": count,
        "scored_questions": len(scored),
        "correct_questions": correct,
        "accuracy": correct / len(scored) if scored else None,
        "average_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds": latency_p95,
        "llm_token_questions": len(llm_token_values),
        "average_llm_tokens": statistics.mean(llm_token_values) if llm_token_values else None,
        "total_llm_tokens": sum(llm_token_values) if llm_token_values else None,
        "average_tool_calls": statistics.mean(tool_calls) if tool_calls else 0,
        "total_tool_calls": sum(tool_calls),
        "failures": failures,
        "failure_rate": failures / count if count else None,
    }


def evaluate_records(records):
    evaluated = []
    for record in records:
        result = dict(record)
        if record.get("success") is False or record.get("status") == "failed":
            answer_eval = {
                "correct": None,
                "method": "operational_failure",
                "normalized_generated": "",
            }
        else:
            answer_eval = evaluate_answer(
                record.get("generated_answer"),
                record.get("expected_answer"),
                record.get("question_type"),
            )
        result["correctness"] = answer_eval["correct"]
        result["evaluation"] = answer_eval
        evaluated.append(result)

    by_pipeline = defaultdict(list)
    by_pipeline_and_type = defaultdict(list)
    for record in evaluated:
        pipeline = record.get("pipeline", "unknown")
        question_type = record.get("question_type", "unknown")
        by_pipeline[pipeline].append(record)
        by_pipeline_and_type[(pipeline, question_type)].append(record)

    return {
        "per_question": evaluated,
        "by_pipeline": {pipeline: _summary(rows) for pipeline, rows in sorted(by_pipeline.items())},
        "by_pipeline_and_type": {
            f"{pipeline}:{question_type}": _summary(rows)
            for (pipeline, question_type), rows in sorted(by_pipeline_and_type.items())
        },
    }


def compute_metrics(records):
    return evaluate_records(records)
