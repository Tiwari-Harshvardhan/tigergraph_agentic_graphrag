import json
import unittest
from unittest.mock import patch

from backend.agent.executor import execute_agent
from backend.llm import ModelToolCall, ModelTurn
from backend.pipelines.agentic_graphrag import run_agentic_graphrag
from backend.pipelines._common import dispatch_tool
from backend.pipelines.graphrag import ANSWER_PROMPT, run_graphrag
from backend.pipelines.rag import run_rag
from backend.tools.event_aggregate import event_aggregate
from backend.tools.event_filter import event_filter
from backend.tools.event_lookup import event_lookup
from backend.tools.graph_context import graph_context
from backend.tools.temporal_search import temporal_search
from backend.tools.vector_search import vector_search
from backend.tools.validation import normalize_numeric_arguments


class FakeModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.turns.pop(0)


class ToolContractTests(unittest.TestCase):
    @staticmethod
    def assert_failure_shape(test_case, result):
        test_case.assertEqual(set(result), {"success", "tool", "error"})
        test_case.assertIs(result["success"], False)
        test_case.assertIsInstance(result["error"], str)

    @staticmethod
    def assert_success_shape(test_case, result, tool):
        test_case.assertEqual(set(result), {"success", "tool", "data"})
        test_case.assertIs(result["success"], True)
        test_case.assertEqual(result["tool"], tool)
        test_case.assertIsInstance(result["data"], dict)

    @patch("backend.tools._common.run_installed_query")
    def test_event_lookup_normalizes_vertex_attributes(self, run_query):
        run_query.return_value = [
            {
                "matched_events": [
                    {
                        "v_id": "Q1",
                        "v_type": "events",
                        "attributes": {"event_id": "Q1", "sport": "Canoeing"},
                    }
                ],
                "editions": [],
            }
        ]

        result = event_lookup("Q1")

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["event"], {"event_id": "Q1", "sport": "Canoeing"})

    @patch("backend.tools._common.run_installed_query")
    def test_event_filter_and_aggregate_shapes(self, run_query):
        run_query.return_value = [{"matched_events": [{"v_id": "Q1", "attributes": {"event_id": "Q1"}}]}]
        filtered = event_filter(sport="Biathlon")
        self.assertEqual(filtered["data"], {"results": [{"event_id": "Q1"}], "count": 1})

        run_query.return_value = [{"operation": "count", "total_count": 5, "competitor_sum": 400}]
        aggregated = event_aggregate("count", sport="Biathlon")
        self.assertEqual(aggregated["data"]["value"], 5)
        self.assertEqual(aggregated["data"]["count"], 5)

    @patch("backend.tools._common.run_installed_query")
    def test_winter_edition_year_is_normalized_to_graph_label(self, run_query):
        run_query.return_value = [{"total_count": 5, "competitor_sum": 413, "competitor_min": 80, "competitor_max": 87, "competitor_avg": 82.6}]
        result = event_aggregate(
            "count", sport="Biathlon", edition="2018", season="Winter",
            competitor_op=">", competitor_value=73,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["value"], 5)
        self.assertEqual(run_query.call_args.args[1]["p_edition"], "2018 Winter Olympics")

    @patch("backend.tools._common.run_installed_query")
    def test_temporal_vector_and_graph_context_normalize(self, run_query):
        run_query.return_value = [{"results": [{"v_id": "Q2", "attributes": {"event_id": "Q2", "year": 2016}}]}]
        temporal = temporal_search("after", reference_event_id="Q1", sport="Canoeing")
        self.assertEqual(temporal["data"]["results"][0]["year"], 2016)

        run_query.return_value = [{"results": [{"v_id": "Q1_c0", "attributes": {"chunk_id": "Q1_c0", "text": "evidence"}}]}]
        vector = vector_search([0.0] * 384)
        self.assertEqual(vector["data"]["results"][0]["text"], "evidence")

        run_query.return_value = [
            {"matched_events": [{"v_id": "Q1", "attributes": {"event_id": "Q1"}}]},
            {"editions": [], "event_documents": [], "event_chunks": []},
        ]
        context = graph_context(event_id="Q1")
        self.assertEqual(context["data"]["event"]["event_id"], "Q1")
        self.assertEqual(context["data"]["chunks"], [])

    @patch("backend.tools._common.run_installed_query")
    def test_query_errors_use_failure_envelope(self, run_query):
        run_query.return_value = ("query failed", "REST-1")
        result = event_lookup("Q1")
        self.assertFalse(result["success"])
        self.assertIn("REST-1", result["error"])
        self.assertFalse(event_filter(competitor_op=">")["success"])

    def test_event_lookup_empty_no_result_and_remote_error(self):
        self.assert_failure_shape(self, event_lookup(""))
        with patch("backend.tools._common.run_installed_query", return_value=[{"matched_events": [], "editions": []}]) as query:
            no_result = event_lookup("Q-absent")
            self.assert_success_shape(self, no_result, "event_lookup")
            self.assertFalse(no_result["data"]["found"])
            self.assertIsNone(no_result["data"]["event"])
            query.return_value = ("offline", "REST-500")
            self.assert_failure_shape(self, event_lookup("Q1"))

    def test_event_filter_malformed_no_result_and_remote_error(self):
        self.assert_failure_shape(self, event_filter(competitor_op=">"))
        with patch("backend.tools._common.run_installed_query", return_value=[{"matched_events": []}]) as query:
            no_result = event_filter(sport="not-a-sport")
            self.assert_success_shape(self, no_result, "event_filter")
            self.assertEqual(no_result["data"], {"results": [], "count": 0})
            query.return_value = ("offline", "REST-500")
            self.assert_failure_shape(self, event_filter())

    def test_event_aggregate_invalid_empty_and_remote_error(self):
        self.assert_failure_shape(self, event_aggregate("nonsense"))
        with patch("backend.tools._common.run_installed_query", return_value=[{"total_count": 0, "competitor_sum": 0, "competitor_min": 18446744073709551615, "competitor_max": 0, "competitor_avg": 0}]) as query:
            no_result = event_aggregate("count", sport="not-a-sport")
            self.assert_success_shape(self, no_result, "event_aggregate")
            self.assertEqual(no_result["data"]["value"], 0)
            query.return_value = ("offline", "REST-500")
            self.assert_failure_shape(self, event_aggregate("count"))

    def test_temporal_invalid_empty_and_remote_error(self):
        self.assert_failure_shape(self, temporal_search("before"))
        with patch("backend.tools._common.run_installed_query", return_value=[{"results": []}]) as query:
            no_result = temporal_search("between", from_year=2012, to_year=2012)
            self.assert_success_shape(self, no_result, "temporal_search")
            self.assertEqual(no_result["data"], {"results": [], "count": 0})
            query.return_value = ("offline", "REST-500")
            self.assert_failure_shape(self, temporal_search("first"))

    def test_vector_invalid_empty_and_remote_error(self):
        self.assert_failure_shape(self, vector_search(None))
        with patch("backend.tools._common.run_installed_query", return_value=[{"results": []}]) as query:
            no_result = vector_search([0.0] * 384)
            self.assert_success_shape(self, no_result, "vector_search")
            self.assertEqual(no_result["data"], {"results": [], "count": 0})
            query.return_value = ("offline", "REST-500")
            self.assert_failure_shape(self, vector_search([0.0] * 384))

    def test_graph_context_invalid_empty_and_remote_error(self):
        self.assert_failure_shape(self, graph_context())
        empty_payload = [{key: [] for key in ["matched_events", "editions", "event_documents", "chunk_documents", "event_chunks", "selected_chunks", "preceding_events"]}]
        with patch("backend.tools._common.run_installed_query", return_value=empty_payload) as query:
            no_result = graph_context(event_id="Q-absent")
            self.assert_success_shape(self, no_result, "graph_context")
            self.assertIsNone(no_result["data"]["event"])
            self.assertEqual(no_result["data"]["chunks"], [])
            query.return_value = ("offline", "REST-500")
            self.assert_failure_shape(self, graph_context(event_id="Q1"))


class ToolArgumentValidationTests(unittest.TestCase):
    @patch("backend.tools._common.run_installed_query")
    def test_numeric_arguments_follow_tool_schema(self, run_query):
        run_query.return_value = [{"total_count": 5, "competitor_sum": 400, "competitor_min": 80, "competitor_max": 87, "competitor_avg": 85.4}]
        integer_result = dispatch_tool(
            "event_aggregate",
            {"operation": "count", "competitor_op": ">", "competitor_value": 73},
            "count events",
        )
        self.assertTrue(integer_result["success"])
        self.assertIs(type(run_query.call_args.args[1]["p_competitor_value"]), int)
        self.assertEqual(run_query.call_args.args[1]["p_competitor_value"], 73)

        string_result = dispatch_tool(
            "event_aggregate",
            {"operation": "count", "competitor_op": ">", "competitor_value": "73"},
            "count events",
        )
        self.assertTrue(string_result["success"])
        self.assertIs(type(run_query.call_args.args[1]["p_competitor_value"]), int)
        self.assertEqual(run_query.call_args.args[1]["p_competitor_value"], 73)

        run_query.reset_mock()
        invalid_result = dispatch_tool(
            "event_aggregate",
            {"operation": "count", "competitor_op": ">", "competitor_value": "seventy-three"},
            "count events",
        )
        self.assertFalse(invalid_result["success"])
        self.assertEqual(invalid_result["error_type"], "validation_error")
        self.assertIn("must be an integer", invalid_result["error"])
        run_query.assert_not_called()

    def test_float_string_coercion_only_applies_to_number_schema(self):
        result = normalize_numeric_arguments(
            "synthetic_tool",
            {"ratio": "73.0", "label": "73"},
            {"ratio": {"type": "number"}, "label": {"type": "string"}},
        )
        self.assertEqual(result, {"ratio": 73.0, "label": "73"})
        with self.assertRaisesRegex(ValueError, "must be a number"):
            normalize_numeric_arguments(
                "synthetic_tool", {"ratio": "seventy-three"}, {"ratio": {"type": "number"}}
            )


class PipelineTests(unittest.TestCase):
    @patch("backend.pipelines.rag.vector_search")
    def test_rag_retrieves_then_answers(self, search):
        search.return_value = {
            "success": True,
            "data": {"results": [{"chunk_id": "Q1_c0", "text": "source text"}], "count": 1},
        }
        model = FakeModel([ModelTurn("Answer [Q1_c0]", prompt_tokens=3, completion_tokens=2)])

        result = run_rag("Question?", model=model, embedder=lambda _: [0.0] * 384)

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["answer"], "Answer [Q1_c0]")
        search.assert_called_once()

    @patch("backend.pipelines.graphrag.dispatch_tool")
    def test_graphrag_uses_llm_plan_then_structured_tool(self, dispatch):
        dispatch.return_value = {"success": True, "tool": "event_aggregate", "data": {"value": 5}}
        model = FakeModel(
            [
                ModelTurn('{"tool":"event_aggregate","arguments":{"operation":"count"}}'),
                ModelTurn("There are 5 events."),
            ]
        )

        result = run_graphrag("How many?", model=model)

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["evidence"]["data"]["value"], 5)
        dispatch.assert_called_once_with("event_aggregate", {"operation": "count"}, "How many?")

    @patch("backend.pipelines.graphrag.dispatch_tool")
    def test_graphrag_preserves_synthetic_filtered_count(self, dispatch):
        dispatch.return_value = {"success": True, "tool": "event_aggregate", "data": {"value": 5, "count": 5}}
        model = FakeModel([
            ModelTurn('{"tool":"event_aggregate","arguments":{"operation":"count","sport":"Biathlon","edition":"2018","season":"Winter","competitor_op":">","competitor_value":73}}'),
            ModelTurn("Five events."),
        ])
        result = run_graphrag("How many qualified?", model=model)
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["evidence"]["data"]["value"], 5)
        self.assertEqual(result["data"]["answer"], "Five events.")

    @patch("backend.tools._common.run_installed_query")
    def test_graphrag_planner_numeric_string_is_normalized_by_shared_tool_contract(self, run_query):
        run_query.return_value = [{"total_count": 5, "competitor_sum": 400, "competitor_min": 80, "competitor_max": 87, "competitor_avg": 85.4}]
        model = FakeModel([
            ModelTurn('{"tool":"event_aggregate","arguments":{"operation":"count","competitor_op":">","competitor_value":"73"}}'),
            ModelTurn("Five events qualified.")
        ])

        result = run_graphrag("Count matching events", model=model)

        self.assertTrue(result["success"])
        parameters = run_query.call_args.args[1]
        self.assertIs(type(parameters["p_competitor_value"]), int)
        self.assertEqual(parameters["p_competitor_value"], 73)

    @patch("backend.pipelines.graphrag.dispatch_tool")
    def test_graphrag_synthesis_omits_operation_metadata_and_rejects_tool_calls(self, dispatch):
        dispatch.return_value = {
            "success": True,
            "tool": "event_aggregate",
            "data": {"results": [{"event": "example", "gold": "Athlete"}], "count": 1},
        }
        model = FakeModel([
            ModelTurn('{"tool":"event_aggregate","arguments":{"operation":"count"}}'),
            ModelTurn(None, [ModelToolCall("unexpected", "event_filter", {"sport": "Athletics"})]),
        ])

        result = run_graphrag("Who won this event?", model=model)

        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["termination_reason"], "model_error")
        self.assertIn("attempted a tool call", result["error"])
        dispatch.assert_called_once()
        synthesis_messages, synthesis_kwargs = model.calls[1]
        self.assertEqual(synthesis_kwargs, {})
        self.assertEqual(synthesis_messages[0], {"role": "system", "content": ANSWER_PROMPT})
        self.assertEqual([message["role"] for message in synthesis_messages], ["system", "user"])
        self.assertIn("Who won this event?", synthesis_messages[1]["content"])
        synthesis_context = synthesis_messages[1]["content"].split("Retrieved evidence (JSON):\n", 1)[1]
        self.assertEqual(json.loads(synthesis_context), dispatch.return_value["data"])
        self.assertNotIn("tool", json.loads(synthesis_context))

    @patch("backend.agent.executor.dispatch_tool")
    def test_agentic_executor_performs_model_selected_tool_call(self, dispatch):
        dispatch.return_value = {"success": True, "tool": "event_lookup", "data": {"event": {"event_id": "Q1"}}}
        model = FakeModel(
            [
                ModelTurn(None, [ModelToolCall("call-1", "event_lookup", {"event_id": "Q1"})]),
                ModelTurn("The event is Q1."),
            ]
        )

        result = execute_agent("Tell me about Q1", model=model)

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["tool_calls"][0]["tool"], "event_lookup")
        dispatch.assert_called_once()

    @patch("backend.pipelines.agentic_graphrag.execute_agent")
    def test_agentic_pipeline_adds_latency(self, execute):
        execute.return_value = {"success": True, "pipeline": "agentic_graphrag", "data": {"answer": "Done"}}
        result = run_agentic_graphrag("Question", model=object())
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["data"]["latency_seconds"], 0)

    @patch("backend.pipelines.rag.vector_search")
    def test_rag_retrieval_failure_retains_latency_tool_and_zero_llm_usage(self, search):
        search.return_value = {"success": False, "error": "TigerGraph unavailable"}
        result = run_rag("Question?", model=object(), embedder=lambda _: [0.0] * 384)
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["total_tokens"], 0)
        self.assertGreaterEqual(result["data"]["latency_seconds"], 0)
        self.assertEqual(result["data"]["tool_calls"][0]["tool"], "vector_search")

    def test_graphrag_planner_failure_has_null_usage_and_latency(self):
        class BrokenModel:
            def complete(self, *_args, **_kwargs):
                raise RuntimeError("planner unavailable")

        result = run_graphrag("Question?", model=BrokenModel())
        self.assertFalse(result["success"])
        self.assertIsNone(result["data"]["total_tokens"])
        self.assertGreaterEqual(result["data"]["latency_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
