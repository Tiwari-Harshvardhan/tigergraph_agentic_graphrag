import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger("olympic_graphrag")
_SECRET_PARTS = ("api_key", "secret", "authorization", "credential", "password")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        run = getattr(record, "run", None)
        if run is not None:
            payload["run"] = redact_secrets(run)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)


def new_run_id() -> str:
    return uuid.uuid4().hex


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secrets(value):
    secret_values = [
        item
        for key, item in os.environ.items()
        if any(part in key.casefold() for part in _SECRET_PARTS) and item
    ]
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(part in key.casefold() for part in _SECRET_PARTS) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        for secret in secret_values:
            value = value.replace(secret, "[REDACTED]")
    return value


def append_jsonl(path, record: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(redact_secrets(record), ensure_ascii=False, default=str) + "\n")


def log_run(record: dict) -> None:
    configure_logging()
    logger.info("pipeline_run", extra={"run": redact_secrets(record)})