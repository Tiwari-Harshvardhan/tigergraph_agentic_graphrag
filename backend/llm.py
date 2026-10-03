import json
import logging
import math
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from backend.observability import redact_secrets


load_dotenv()
logger = logging.getLogger("olympic_graphrag.llm")


@dataclass
class ModelToolCall:
    id: str
    name: str
    arguments: dict
    thought_signature: bytes | None = None


@dataclass
class ModelTurn:
    content: str | None
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class OpenAICompatibleChatModel:
    def __init__(self, client, model: str, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    def complete(self, messages, *, tools=None, json_mode=False) -> ModelTurn:
        request = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        else:
            # Make synthesis/planner calls explicitly non-tool-capable. Some
            # OpenAI-compatible routers otherwise let tool-shaped generations
            # escape despite having no tools in the request.
            request["tool_choice"] = "none"
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        request_type = "tool" if tools else "json" if json_mode else "chat"
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as error:
            status = getattr(error, "status_code", None) or getattr(error, "http_status", None)
            error_type = type(error).__name__
            detail = redact_secrets(str(error))
            logger.error(
                "llm_request_failed provider=%s model=%s request_type=%s latency_seconds=%.3f status=%s error_type=%s error=%s",
                os.getenv("LLM_PROVIDER", "openai-compatible"), self.model, request_type,
                time.perf_counter() - started, status, error_type, detail,
            )
            raise RuntimeError(
                f"LLM request failed: provider={os.getenv('LLM_PROVIDER', 'openai-compatible')} "
                f"model={self.model} request_type={request_type} status={status} "
                f"error_type={error_type} error={detail}"
            ) from error
        message = response.choices[0].message
        tool_calls = []
        for tool_call in message.tool_calls or []:
            arguments = json.loads(tool_call.function.arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError(f"Tool arguments must be an object: {tool_call.function.name}")
            tool_calls.append(
                ModelToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments,
                )
            )

        usage = response.usage
        logger.info(
            "llm_request_complete provider=%s model=%s request_type=%s latency_seconds=%.3f prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            os.getenv("LLM_PROVIDER", "openai-compatible"), self.model, request_type,
            time.perf_counter() - started, getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None), getattr(usage, "total_tokens", None),
        )
        return ModelTurn(
            content=message.content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )


class GeminiChatModel:
    def __init__(self, client, model: str, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _convert_messages(messages, types):
        system_parts = []
        contents = []
        tool_names = {}

        for message in messages:
            role = message.get("role")
            if role == "system":
                if message.get("content"):
                    system_parts.append(message["content"])
                continue

            if role == "assistant" and message.get("tool_calls"):
                parts = []
                for call in message["tool_calls"]:
                    function = call.get("function", {})
                    name = function.get("name", "")
                    call_id = call.get("id", name)
                    tool_names[call_id] = name
                    arguments = function.get("arguments", "{}")
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    thought_sig = call.get("thought_signature")
                    if thought_sig is not None:
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(name=name, args=arguments),
                                thought_signature=thought_sig,
                            )
                        )
                    else:
                        parts.append(types.Part.from_function_call(name=name, args=arguments))
                contents.append(types.Content(role="model", parts=parts))
                continue

            if role == "tool":
                call_id = message.get("tool_call_id", "")
                name = tool_names.get(call_id)
                if not name:
                    raise ValueError(f"No Gemini function call found for tool response {call_id!r}")
                response = message.get("content", "{}")
                response = json.loads(response) if isinstance(response, str) else response
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(name=name, response={"result": response})],
                    )
                )
                continue

            content = message.get("content") or ""
            converted_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=converted_role, parts=[types.Part.from_text(text=content)]))

        return "\n\n".join(system_parts) or None, contents

    def complete(self, messages, *, tools=None, json_mode=False) -> ModelTurn:
        from google.genai import types

        system_instruction, contents = self._convert_messages(messages, types)
        config = {"temperature": self.temperature}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if tools:
            declarations = []
            for tool in tools:
                function = tool.get("function", tool)
                declarations.append(
                    types.FunctionDeclaration(
                        name=function["name"],
                        description=function.get("description"),
                        parameters_json_schema=function.get("parameters", {"type": "object"}),
                    )
                )
            config["tools"] = [types.Tool(function_declarations=declarations)]
        if json_mode:
            config["response_mime_type"] = "application/json"

        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config),
                )
                break
            except Exception as error:
                is_transient = any(
                    code in str(error)
                    for code in ("503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE")
                )
                if is_transient and attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                    continue
                raise

        candidate = (response.candidates or [None])[0]
        parts = candidate.content.parts if candidate and candidate.content else []
        text_parts = []
        tool_calls = []
        for part_index, part in enumerate(parts):
            if part.text:
                text_parts.append(part.text)
            function_call = part.function_call
            if function_call and function_call.name:
                arguments = dict(function_call.args or {})
                tool_calls.append(
                    ModelToolCall(
                        id=f"gemini-{part_index}-{function_call.name}",
                        name=function_call.name,
                        arguments=arguments,
                        thought_signature=getattr(part, "thought_signature", None),
                    )
                )

        usage = response.usage_metadata
        return ModelTurn(
            content="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )


def get_model_config() -> dict:
    provider = os.getenv("LLM_PROVIDER", "gemini" if os.getenv("GEMINI_API_KEY") else "openai").lower()
    if provider == "huggingface":
        model_name = os.getenv("HF_MODEL") or "openai/gpt-oss-120b:fastest"
    elif provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL") or "llama3.1:8b-instruct-q4_K_M"
    elif provider == "gemini":
        model_name = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-3.8-flash"
    else:
        model_name = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    except ValueError as error:
        raise RuntimeError("LLM_TEMPERATURE must be a number between 0 and 2") from error
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise RuntimeError("LLM_TEMPERATURE must be a number between 0 and 2")
    return {"provider": provider, "model": model_name, "temperature": temperature}


def get_chat_model():
    model_config = get_model_config()
    provider = model_config["provider"]
    if provider == "huggingface":
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("Set HF_TOKEN locally to enable Hugging Face chat generation")

        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("HF_BASE_URL") or "https://router.huggingface.co/v1",
        )
        return OpenAICompatibleChatModel(client, model_config["model"], model_config["temperature"])

    if provider == "ollama":
        from openai import OpenAI

        base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        client = OpenAI(api_key="ollama-local", base_url=base_url)
        return OpenAICompatibleChatModel(client, model_config["model"], model_config["temperature"])

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY to enable Gemini answer generation")

        from google import genai

        client = genai.Client(api_key=api_key)
        return GeminiChatModel(client, model_config["model"], model_config["temperature"])

    if provider not in {"openai", "openai-compatible"}:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider!r}")

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or LLM_API_KEY to enable answer generation")

    from openai import OpenAI

    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return OpenAICompatibleChatModel(client, model_config["model"], model_config["temperature"])
