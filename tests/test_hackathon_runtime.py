import json
import os
import tempfile
import unittest
import urllib.request
import httpx
from openai import OpenAI
from http.server import ThreadingHTTPServer
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.agent.executor import execute_agent
from backend.llm import GeminiChatModel, ModelToolCall, ModelTurn, OpenAICompatibleChatModel, get_chat_model, get_model_config
from backend.observability import append_jsonl
from backend.services.tigergraph_service import REQUIRED_EDGES, REQUIRED_QUERIES, REQUIRED_VERTICES, TigerGraphService
from app.server import DemoHandler
from benchmark.run import DEFAULT_INPUTS, _run_one, load_questions, run_benchmark
from evaluation.answers import evaluate_answer
from evaluation.metrics import evaluate_records


class FakeModel:
    def __init__(self, turns):
        self.turns = list(turns)

    def complete(self, messages, **kwargs):
        return self.turns.pop(0)


class FakeGeminiModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeConnection:
    def __init__(self, reachable=True, schema=None, queries=None):
        self.reachable = reachable
        self.schema = schema or {
            "GraphName": "wikipedia_corpus",
            "VertexTypes": [
                {"Name": name, "EmbeddingAttributes": [{"Name": "embedding", "Dimension": 384}] if name == "chunk" else []}
                for name in REQUIRED_VERTICES
            ],
            "EdgeTypes": [
                {"Name": name, "FromVertexTypeName": "events", "ToVertexTypeName": "events"}
                for name in REQUIRED_EDGES
            ],
        }
        self.queries = queries or {
            f"GET /query/wikipedia_corpus/{name}": {} for name in REQUIRED_QUERIES
        }

    def echo(self):
        if not self.reachable:
            raise ConnectionError("offline")
        return "Hello GSQL"

    def getSchema(self):
        return self.schema

    def getInstalledQueries(self):
        return self.queries


class GeminiAdapterTests(unittest.TestCase):
    def test_native_function_call_and_provider_usage(self):
        function_call = SimpleNamespace(name="event_lookup", args={"event_id": "Q1"})
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=None, function_call=function_call)]))],
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=4, total_token_count=14),
        )
        models = FakeGeminiModels(response)
        model = GeminiChatModel(SimpleNamespace(models=models), "test-model")
        turn = model.complete(
            [{"role": "system", "content": "Use tools"}, {"role": "user", "content": "Find Q1"}],
            tools=[{"type": "function", "function": {"name": "event_lookup", "parameters": {"type": "object"}}}],
        )
        self.assertEqual(turn.tool_calls[0].arguments, {"event_id": "Q1"})
        self.assertEqual((turn.prompt_tokens, turn.completion_tokens, turn.total_tokens), (10, 4, 14))
        self.assertEqual(models.calls[0]["config"].tools[0].function_declarations[0].name, "event_lookup")

    def test_function_response_round_trip(self):
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="done", function_call=None)]))], usage_metadata=None)
        model = GeminiChatModel(SimpleNamespace(models=FakeGeminiModels(response)), "test-model")
        model.complete(
            [
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "event_lookup", "arguments": "{\"event_id\":\"Q1\"}"}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": "{\"success\":true}"},
            ]
        )
        self.assertIsNone(model.client.models.calls[0]["config"].system_instruction)
        self.assertIsNone(model.client.models.calls[0]["contents"][0].parts[0].text)

    def test_absent_usage_is_not_reported_as_zero(self):
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="answer", function_call=None)]))], usage_metadata=None)
        model = GeminiChatModel(SimpleNamespace(models=FakeGeminiModels(response)), "test-model")
        turn = model.complete([{"role": "user", "content": "question"}])
        self.assertIsNone(turn.prompt_tokens)
        self.assertIsNone(turn.completion_tokens)
        self.assertIsNone(turn.total_tokens)

    def test_missing_gemini_key_is_actionable(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                get_chat_model()

    def test_provider_model_and_temperature_are_reproducible(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-test", "LLM_TEMPERATURE": "0.2"},
        ):
            self.assertEqual(
                get_model_config(),
                {"provider": "gemini", "model": "gemini-test", "temperature": 0.2},
            )


class OpenAICompatibleAdapterTests(unittest.TestCase):
    def test_provider_402_is_not_retried_by_openai_sdk(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(402, json={"error": {"message": "credits exhausted"}})

        client = OpenAI(
            api_key="offline-test-key",
            base_url="https://offline.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        model = OpenAICompatibleChatModel(client, "test-model")
        try:
            with self.assertRaisesRegex(RuntimeError, "status=402"):
                model.complete([{"role": "user", "content": "question"}])
        finally:
            client.close()
        self.assertEqual(len(requests), 1)

    def test_tool_choice_none_is_serialized_without_request_state_leakage(self):
        captured = []

        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            })

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenAI(api_key="offline-test-key", base_url="https://offline.invalid/v1", http_client=http_client)
        model = OpenAICompatibleChatModel(client, "test-model")
        tool_schema = [{"type": "function", "function": {"name": "offline_tool", "parameters": {"type": "object"}}}]
        try:
            model.complete([{"role": "user", "content": "tool-capable request"}], tools=tool_schema)
            model.complete([{"role": "system", "content": "evidence only"}, {"role": "user", "content": "answer"}])
        finally:
            client.close()

        self.assertEqual(captured[0]["tool_choice"], "auto")
        self.assertIn("tools", captured[0])
        self.assertEqual(captured[1]["tool_choice"], "none")
        self.assertNotIn("tools", captured[1])
        self.assertNotIn("response_format", captured[1])
        self.assertEqual(captured[1]["temperature"], 0.0)
        self.assertEqual(captured[1]["messages"], [
            {"role": "system", "content": "evidence only"},
            {"role": "user", "content": "answer"},
        ])

    def test_chat_completion_and_multi_turn_tool_call(self):
        responses = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Hello.", tool_calls=None))],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(name="event_lookup", arguments='{"event_id":"Q1"}'),
                )]))],
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, total_tokens=7),
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Found Q1.", tool_calls=None))],
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10),
            ),
        ]
        completions = SimpleNamespace(create=unittest.mock.Mock(side_effect=responses))
        model = OpenAICompatibleChatModel(SimpleNamespace(chat=SimpleNamespace(completions=completions)), "test-model")
        tools = [{"type": "function", "function": {"name": "event_lookup", "parameters": {"type": "object"}}}]

        chat_turn = model.complete([{"role": "user", "content": "Say hello"}])
        tool_turn = model.complete([{"role": "user", "content": "Find Q1"}], tools=tools)
        final_turn = model.complete([
            {"role": "user", "content": "Find Q1"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "event_lookup", "arguments": '{"event_id":"Q1"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": '{"success":true}'},
        ])

        self.assertEqual(chat_turn.content, "Hello.")
        self.assertEqual(tool_turn.tool_calls[0].arguments, {"event_id": "Q1"})
        self.assertEqual((tool_turn.prompt_tokens, tool_turn.completion_tokens, tool_turn.total_tokens), (4, 3, 7))
        self.assertEqual(final_turn.content, "Found Q1.")
        self.assertEqual(completions.create.call_count, 3)
        self.assertEqual(completions.create.call_args_list[0].kwargs["tool_choice"], "none")
        self.assertEqual(completions.create.call_args_list[1].kwargs["tool_choice"], "auto")
        self.assertEqual(completions.create.call_args_list[2].kwargs["tool_choice"], "none")

    def test_huggingface_config_and_missing_token(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "huggingface",
            "HF_MODEL": "openai/gpt-oss-120b:fastest",
        }, clear=True):
            self.assertEqual(get_model_config()["model"], "openai/gpt-oss-120b:fastest")
            with self.assertRaisesRegex(RuntimeError, "Set HF_TOKEN locally"):
                get_chat_model()

    def test_ollama_provider_config_and_local_base_url(self):
        target_model = "llama3.1:8b-instruct-q4_K_M"
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": target_model,
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "LLM_TEMPERATURE": "0",
        }, clear=True), patch("openai.OpenAI") as openai_client:
            model_config = get_model_config()
            model = get_chat_model()

        self.assertEqual(model_config, {"provider": "ollama", "model": target_model, "temperature": 0.0})
        self.assertEqual(model.model, target_model)
        openai_client.assert_called_once_with(api_key="ollama-local", base_url="http://localhost:11434/v1")

    def test_ollama_defaults_to_local_service_and_target_model(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            self.assertEqual(get_model_config()["model"], "llama3.1:8b-instruct-q4_K_M")


class TigerGraphServiceTests(unittest.TestCase):
    def test_live_service_verifies_graph_contract(self):
        service = TigerGraphService(connection_factory=lambda: FakeConnection(), sleep=lambda _: None)
        result = service.ensure_ready()
        self.assertTrue(result["ready"])
        self.assertEqual(set(result["queries"]), REQUIRED_QUERIES)

    def test_unavailable_graph_reports_manual_start(self):
        service = TigerGraphService(connection_factory=lambda: FakeConnection(reachable=False), sleep=lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "Start the wikipedia_corpus Savanna workspace manually"):
            service.ensure_ready()

    def test_auto_start_does_not_claim_unsupported_capability(self):
        service = TigerGraphService(connection_factory=lambda: FakeConnection(reachable=False), sleep=lambda _: None)
        service.auto_start = True
        with self.assertRaisesRegex(RuntimeError, "not available through the configured"):
            service.ensure_ready()

    def test_start_is_idempotent_when_graph_is_already_up(self):
        service = TigerGraphService(connection_factory=lambda: FakeConnection(), sleep=lambda _: None)
        self.assertTrue(service.start()["ready"])

    def test_wait_times_out(self):
        service = TigerGraphService(connection_factory=lambda: FakeConnection(reachable=False), sleep=lambda _: None)
        with self.assertRaises(TimeoutError):
            service.wait_until_ready(timeout=0)


class BenchmarkAndEvaluationTests(unittest.TestCase):
    @staticmethod
    def _write_generic_questions(path):
        records = [
            {"qid": "q-alpha", "question": "Question alpha?", "qtype": "lookup", "answer": "Alpha"},
            {"qid": "q-beta", "question": "Question beta?", "qtype": "lookup", "answer": "Beta"},
            {"qid": "q-gamma", "question": "Question gamma?", "qtype": "lookup", "answer": "Gamma"},
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_benchmark_stops_at_provider_402_without_scoring_or_consuming_next_question(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "questions.jsonl"
            output = Path(temp) / "results.jsonl"
            self._write_generic_questions(source)
            calls = []

            def depleted(question, **_kwargs):
                calls.append(question)
                return {
                    "success": False,
                    "error": "LLM request failed: provider=huggingface model=test-model request_type=chat status=402 error_type=APIStatusError error=credit exhausted",
                    "data": {"answer": None, "termination_reason": "model_error", "tool_calls": []},
                }

            with patch.dict("benchmark.run.PIPELINES", {"rag": depleted}), patch("benchmark.run.log_run"), patch(
                "benchmark.run.time.sleep"
            ):
                summary = run_benchmark(
                    pipelines=["rag"], input_paths=[source], output_path=output, ensure_ready=False
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(calls, ["Question alpha?"])
            self.assertEqual(summary["result_count"], 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["question_id"], rows[0]["pipeline"]), ("q-alpha", "rag"))
            self.assertEqual((rows[0]["error_status"], rows[0]["failure_category"]), ("402", "provider_credit"))
            self.assertIsNone(rows[0]["correctness"])
            self.assertEqual(rows[0]["termination_reason"], "model_error")
            evaluated = evaluate_records(rows)
            self.assertEqual(evaluated["by_pipeline"]["rag"]["scored_questions"], 0)
            self.assertEqual(evaluated["by_pipeline"]["rag"]["failures"], 1)

    def test_parallel_benchmark_mode_is_rejected_before_any_execution(self):
        with patch("benchmark.run.ensure_tigergraph_ready") as ready:
            with self.assertRaisesRegex(ValueError, "immediate stop"):
                run_benchmark(pipelines=["rag"], parallel=True)
        ready.assert_not_called()

    def test_benchmark_resume_is_append_only_and_skips_completed_generic_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "questions.jsonl"
            output = Path(temp) / "results.jsonl"
            self._write_generic_questions(source)
            first_calls = []

            def first_attempt(question, **_kwargs):
                first_calls.append(question)
                if question == "Question beta?":
                    return {"success": False, "error": "temporary failure", "data": {"answer": None}}
                return {"success": True, "data": {"answer": "Alpha", "tool_calls": [], "evidence": []}}

            with patch.dict("benchmark.run.PIPELINES", {"rag": first_attempt}), patch("benchmark.run.log_run"), patch(
                "benchmark.run.time.sleep"
            ):
                initial = run_benchmark(
                    pipelines=["rag"], input_paths=[source], output_path=output, ensure_ready=False
                )
            original_bytes = output.read_bytes()
            with patch.dict("benchmark.run.PIPELINES", {"rag": lambda *_args, **_kwargs: self.fail("must not overwrite by rerunning")}), patch(
                "benchmark.run.ensure_tigergraph_ready"
            ):
                with self.assertRaises(FileExistsError):
                    run_benchmark(pipelines=["rag"], input_paths=[source], output_path=output, ensure_ready=False)
            self.assertEqual(output.read_bytes(), original_bytes)

            resumed_calls = []

            def resumed(question, **_kwargs):
                resumed_calls.append(question)
                return {"success": True, "data": {"answer": "Beta", "tool_calls": [], "evidence": []}}

            with patch.dict("benchmark.run.PIPELINES", {"rag": resumed}), patch("benchmark.run.log_run"), patch(
                "benchmark.run.time.sleep"
            ):
                resumed_summary = run_benchmark(
                    pipelines=["rag"], input_paths=[source], output_path=output, resume=True, ensure_ready=False
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(first_calls, ["Question alpha?", "Question beta?", "Question gamma?"])
            self.assertEqual(resumed_calls, ["Question beta?"])
            self.assertEqual(initial["result_count"], 3)
            self.assertEqual(resumed_summary["result_count"], 1)
            self.assertEqual(len(rows), 4)
            self.assertEqual(output.read_bytes().splitlines()[0], original_bytes.splitlines()[0])
            self.assertEqual([(r["question_id"], r["pipeline"]) for r in rows[:3]], [
                ("q-alpha", "rag"), ("q-beta", "rag"), ("q-gamma", "rag")
            ])

    def test_all_question_records_load_without_exposing_hidden_answers(self):
        questions = load_questions(DEFAULT_INPUTS)
        self.assertEqual(len(questions), 150)
        hidden = load_questions([Path("questions/eval_hidden.jsonl")])
        self.assertEqual(len(hidden), 50)
        self.assertTrue(all("answer" not in record for record in hidden))

    def test_benchmark_resume_and_correctness(self):
        question = load_questions([Path("questions/eval_public.jsonl")], limit=1)[0]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "results.jsonl"
            fake = lambda question, **kwargs: {"success": True, "data": {"answer": "5", "token_usage": {"total_tokens": None}, "tool_calls": [], "evidence": []}}
            with patch.dict("benchmark.run.PIPELINES", {"rag": fake}), patch("benchmark.run.ensure_tigergraph_ready"):
                first = run_benchmark(pipelines=["rag"], input_paths=["questions/eval_public.jsonl"], output_path=output, limit=1)
                second = run_benchmark(pipelines=["rag"], input_paths=["questions/eval_public.jsonl"], output_path=output, limit=1, resume=True)
            self.assertEqual(first["result_count"], 1)
            self.assertEqual(second["result_count"], 0)
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(row["token_usage"]["total_tokens"])

    def test_benchmark_resume_retries_failed_question(self):
        calls = {"count": 0}

        def flaky(question, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return {"success": False, "error": "temporary failure", "data": {}}
            return {"success": True, "data": {"answer": "done", "tool_calls": [], "evidence": []}}

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "retry.jsonl"
            with patch.dict("benchmark.run.PIPELINES", {"rag": flaky}), patch("benchmark.run.ensure_tigergraph_ready"), patch(
                "benchmark.run.log_run"
            ):
                first = run_benchmark(pipelines=["rag"], input_paths=["questions/eval_public.jsonl"], output_path=output, limit=1)
                second = run_benchmark(pipelines=["rag"], input_paths=["questions/eval_public.jsonl"], output_path=output, limit=1, resume=True)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual((first["result_count"], second["result_count"]), (1, 1))
            self.assertEqual([row["success"] for row in rows], [False, True])

    def test_all_three_pipelines_can_cover_all_150_records_offline(self):
        def fake(question, **kwargs):
            return {"success": True, "data": {"answer": "offline", "tool_calls": [], "evidence": []}}

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "all-runs.jsonl"
            fake_pipelines = {name: fake for name in ("rag", "graphrag", "agentic")}
            with patch.dict("benchmark.run.PIPELINES", fake_pipelines), patch("benchmark.run.log_run"), patch(
                "benchmark.run.ensure_tigergraph_ready"
            ):
                summary = run_benchmark(
                    pipelines=["rag", "graphrag", "agentic"],
                    input_paths=DEFAULT_INPUTS,
                    output_path=output,
                    ensure_ready=True,
                )
            self.assertEqual(summary["question_count"], 150)
            self.assertEqual(summary["result_count"], 450)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 450)
            combinations = {(row["pipeline"], row["question_id"]) for row in rows}
            self.assertEqual(len(combinations), 450)
            hidden_rows = [row for row in rows if row["question_id"].startswith("hid-")]
            self.assertTrue(all(row["expected_answer"] is None for row in hidden_rows))
            self.assertTrue(all(row["correctness"] is None for row in hidden_rows))

    def test_evaluator_handles_numeric_names_lists_and_unscored(self):
        self.assertTrue(evaluate_answer("There were five events.", "5", "aggregation")["correct"])
        self.assertTrue(evaluate_answer("Five events were contested by 73 competitors.", "5", "aggregation")["correct"])
        self.assertFalse(evaluate_answer("Four events were contested by 73 competitors.", "5", "aggregation")["correct"])
        self.assertTrue(evaluate_answer("Emilie Fer", "Émilie Fer", "lookup")["correct"])
        self.assertTrue(evaluate_answer("Men’s marathon", "Athletics 2008 Olympics Men’s marathon", "superlative")["correct"])
        self.assertTrue(evaluate_answer("beta and alpha", ["alpha", "beta"], "multi_hop")["correct"])
        self.assertIsNone(evaluate_answer("some answer", None)["correct"])

    def test_benchmark_rows_retain_runtime_and_failure_diagnostics(self):
        record = {"_question_id": "pub-002", "question": "question", "answer": "5", "qtype": "aggregation"}
        with patch("benchmark.run.get_model_config", return_value={"provider": "huggingface", "model": "openai/gpt-oss-20b:fastest", "temperature": 0.0}):
            with patch("benchmark.run._pipeline_call", return_value={"success": True, "data": {
                "answer": "Five events", "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
                "tool_calls": [{"tool": "event_filter"}], "termination_reason": "final_answer",
            }}):
                success = _run_one((record, "agentic", "run", "time", 5, 6))
            with patch("benchmark.run._pipeline_call", return_value={"success": False,
                "error": "LLM request failed: provider=huggingface model=openai/gpt-oss-20b:fastest request_type=chat status=400 error_type=BadRequestError error=Tool failure",
                "data": {"tool_calls": [{"tool": "vector_search"}], "termination_reason": "model_error"}}):
                failed = _run_one((record, "graphrag", "run", "time", 5, 6))
        self.assertEqual((success["provider"], success["model"], success["termination_reason"]),
                         ("huggingface", "openai/gpt-oss-20b:fastest", "final_answer"))
        self.assertEqual(success["status"], "success")
        self.assertEqual((success["prompt_tokens"], success["completion_tokens"], success["total_tokens"]), (12, 8, 20))
        self.assertEqual((failed["error_type"], failed["error_status"], failed["diagnostic"]["request_type"]),
                         ("BadRequestError", "400", "chat"))
        self.assertEqual(failed["status"], "failed")

    def test_metrics_do_not_invent_tokens(self):
        result = evaluate_records([
            {"pipeline": "rag", "question_type": "aggregation", "expected_answer": "5", "generated_answer": "5", "latency_seconds": 0.5, "token_usage": {"total_tokens": None}, "tool_call_count": 1, "success": True},
            {"pipeline": "rag", "question_type": "lookup", "expected_answer": None, "generated_answer": "not scored", "latency_seconds": 1.0, "token_usage": {"total_tokens": None}, "tool_call_count": 0, "success": True},
        ])
        summary = result["by_pipeline"]["rag"]
        self.assertEqual(summary["scored_questions"], 1)
        self.assertIsNone(summary["total_llm_tokens"])
        self.assertEqual(summary["total_tool_calls"], 1)

    def test_jsonl_writer_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "trace.jsonl"
            append_jsonl(output, {"api_key": "do-not-write", "data": {"answer": "ok"}})
            serialized = output.read_text(encoding="utf-8")
            self.assertNotIn("do-not-write", serialized)
            self.assertIn("[REDACTED]", serialized)


class AgentTrajectoryTests(unittest.TestCase):
    def test_agent_uses_same_model_instance_for_planning_and_final_answer(self):
        model = FakeModel([
            ModelTurn(None, [ModelToolCall("call-1", "event_lookup", {"event_id": "Q1"})]),
            ModelTurn("Found Q1."),
        ])
        with patch("backend.agent.executor.dispatch_tool", return_value={"success": True, "data": {"event": {"event_id": "Q1"}}}):
            result = execute_agent("Find Q1", model=model)
        self.assertTrue(result["success"])
        self.assertEqual(len(model.turns), 0)

    @patch("backend.agent.executor.dispatch_tool")
    def test_max_tool_turns_preserves_partial_trace(self, dispatch):
        dispatch.return_value = {"success": True, "tool": "event_lookup", "data": {"event": {"event_id": "Q1"}}}
        model = FakeModel([ModelTurn(None, [ModelToolCall("call-1", "event_lookup", {"event_id": "Q1"})])])

        result = execute_agent("Find Q1", model=model, max_tool_turns=1)

        self.assertFalse(result["success"])
        self.assertEqual(len(result["data"]["tool_calls"]), 1)
        self.assertEqual(len(result["data"]["evidence"]), 1)
        self.assertEqual(result["data"]["termination_reason"], "max_tool_turns")
        self.assertIn("Maximum tool turns", result["error"])

    @patch("backend.agent.executor.dispatch_tool")
    def test_tool_failure_is_retained_and_agent_can_continue(self, dispatch):
        dispatch.return_value = {"success": False, "tool": "event_lookup", "error": "missing event"}
        model = FakeModel([
            ModelTurn(None, [ModelToolCall("call-1", "event_lookup", {"event_id": "Q404"})]),
            ModelTurn("No matching event was found."),
        ])

        result = execute_agent("Find Q404", model=model)

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["errors"][0]["error"], "missing event")
        self.assertEqual(result["data"]["tool_calls"][0]["result"]["success"], False)
        self.assertEqual(result["data"]["termination_reason"], "final_answer")

    @patch("backend.agent.executor.dispatch_tool")
    def test_agent_recovers_from_tool_failure_with_a_later_structured_call(self, dispatch):
        dispatch.side_effect = [
            {"success": False, "tool": "event_filter", "error_type": "validation_error", "error": "competitor_value must be an integer"},
            {"success": True, "tool": "event_aggregate", "data": {"value": 5}},
        ]
        model = FakeModel([
            ModelTurn(None, [ModelToolCall("call-1", "event_filter", {"competitor_value": "seventy-three"})]),
            ModelTurn(None, [ModelToolCall("call-2", "event_aggregate", {"operation": "count", "competitor_value": 73})]),
            ModelTurn("Five events qualified.")
        ])

        result = execute_agent("Count matching events", model=model)

        self.assertTrue(result["success"])
        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual(result["data"]["errors"][0]["error_type"], "validation_error")
        self.assertEqual(result["data"]["tool_calls"][1]["result"]["data"]["value"], 5)
        self.assertEqual(result["data"]["answer"], "Five events qualified.")

    @patch("backend.agent.executor.dispatch_tool")
    def test_agent_rejects_json_text_that_describes_a_tool_call(self, dispatch):
        model = FakeModel([
            ModelTurn(None, [ModelToolCall("call-1", "event_filter", {"competitor_value": "seventy-three"})]),
            ModelTurn('I should retry this: {"name":"event_aggregate","parameters":{"operation":"count","competitor_value":73}}'),
        ])
        dispatch.return_value = {"success": False, "tool": "event_filter", "error_type": "validation_error", "error": "competitor_value must be an integer"}

        result = execute_agent("Count matching events", model=model)

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["termination_reason"], "invalid_tool_call_text")
        self.assertEqual(result["data"]["errors"][-1]["error_type"], "invalid_tool_call_text")
        self.assertEqual(len(result["data"]["tool_calls"]), 1)
        dispatch.assert_called_once()

    def test_model_error_retains_question_and_unknown_usage(self):
        class BrokenModel:
            def complete(self, *_args, **_kwargs):
                raise RuntimeError("provider unavailable")

        result = execute_agent("Keep this question in trace", model=BrokenModel())
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["original_question"], "Keep this question in trace")
        self.assertEqual(result["data"]["termination_reason"], "model_error")
        self.assertIsNone(result["data"]["prompt_tokens"])

    @patch("backend.agent.executor.dispatch_tool")
    def test_unexecuted_model_call_is_recorded_at_call_limit(self, dispatch):
        dispatch.return_value = {"success": True, "data": {}}
        model = FakeModel([
            ModelTurn(None, [
                ModelToolCall("call-1", "event_lookup", {"event_id": "Q1"}),
                ModelToolCall("call-2", "event_lookup", {"event_id": "Q2"}),
            ])
        ])
        result = execute_agent("Look up two", model=model, max_tool_calls=1)
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["termination_reason"], "max_tool_calls")
        self.assertEqual(len(result["data"]["tool_calls"]), 2)
        self.assertFalse(result["data"]["tool_calls"][1]["executed"])
        dispatch.assert_called_once()

    @patch("backend.agent.executor.dispatch_tool")
    def test_agent_can_make_two_calls_in_separate_turns(self, dispatch):
        dispatch.side_effect = [
            {"success": True, "tool": "event_lookup", "data": {"event": {"event_id": "Q1", "year": 2012}}},
            {"success": True, "tool": "temporal_search", "data": {"results": [{"event_id": "Q2", "year": 2008}]}},
        ]
        model = FakeModel([
            ModelTurn(None, [ModelToolCall("call-1", "event_lookup", {"event_id": "Q1"})], 5, 2, 7),
            ModelTurn(None, [ModelToolCall("call-2", "temporal_search", {"mode": "before", "reference_event_id": "Q1"})], 6, 2, 8),
            ModelTurn("The earlier event is Q2.", prompt_tokens=7, completion_tokens=4, total_tokens=11),
        ])

        result = execute_agent("Find an earlier event", model=model)

        self.assertTrue(result["success"])
        self.assertEqual(len(result["data"]["tool_calls"]), 2)
        self.assertEqual(result["data"]["tool_calls"][1]["result"]["data"]["results"][0]["event_id"], "Q2")
        self.assertEqual(result["data"]["total_tokens"], 26)

    @patch("backend.agent.executor.dispatch_tool")
    def test_agent_recovers_from_malformed_arguments(self, dispatch):
        dispatch.side_effect = [TypeError("unexpected argument"), {"success": True, "data": {"value": 5}}]
        model = FakeModel([
            ModelTurn(None, [ModelToolCall("call-1", "event_aggregate", {"invalid": True})]),
            ModelTurn(None, [ModelToolCall("call-2", "event_aggregate", {"operation": "count"})]),
            ModelTurn("There are 5 events."),
        ])

        result = execute_agent("Count matching events", model=model)

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["errors"][0]["error"], "unexpected argument")
        self.assertEqual(len(result["data"]["tool_calls"]), 2)

    @patch("backend.agent.executor.dispatch_tool")
    def test_agent_accepts_final_answer_without_tool(self, dispatch):
        model = FakeModel([ModelTurn("No relevant event was found.", prompt_tokens=4, completion_tokens=5, total_tokens=9)])
        result = execute_agent("Unknown event", model=model)
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["tool_calls"], [])
        dispatch.assert_not_called()

    @patch("backend.agent.executor.dispatch_tool")
    def test_unknown_tool_is_reported_and_agent_can_recover(self, dispatch):
        dispatch.side_effect = [
            {"success": False, "tool": "made_up", "error": "Unknown tool"},
            {"success": True, "tool": "event_lookup", "data": {"event": {"event_id": "Q1"}}},
        ]
        model = FakeModel([
            ModelTurn(None, [ModelToolCall("call-1", "made_up", {})]),
            ModelTurn(None, [ModelToolCall("call-2", "event_lookup", {"event_id": "Q1"})]),
            ModelTurn("Found Q1."),
        ])
        result = execute_agent("Lookup Q1", model=model)
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["errors"][0]["tool"], "made_up")
        self.assertEqual(len(result["data"]["tool_calls"]), 2)


class DemoServerTests(unittest.TestCase):
    def test_question_ui_and_compare_api(self):
        def fake(question, **kwargs):
            return {"success": True, "data": {"answer": f"answer for {question}", "tool_calls": [], "evidence": []}}

        server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with patch("app.server.ensure_tigergraph_ready", return_value={"ready": True}), patch.dict(
                "app.server.PIPELINES", {"rag": fake, "graphrag": fake, "agentic": fake}
            ):
                page = urllib.request.urlopen(base_url + "/").read().decode("utf-8")
                self.assertIn("Olympic knowledge, tested.", page)
                request = urllib.request.Request(
                    base_url + "/api/run",
                    data=json.dumps({"question": "test", "mode": "compare"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                result = json.loads(urllib.request.urlopen(request).read())
                self.assertEqual(set(result["results"]), {"rag", "graphrag", "agentic"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
