import math
import re


class NumericArgumentError(ValueError):
    """A schema-declared numeric argument could not be safely normalized."""


def normalize_numeric_arguments(tool_name, arguments, properties):
    """Coerce only numeric fields declared by a tool's JSON schema."""
    normalized = dict(arguments)
    for name, schema in properties.items():
        if name not in normalized:
            continue
        expected = schema.get("type")
        value = normalized[name]
        if expected not in {"integer", "number"}:
            continue

        if isinstance(value, bool):
            raise NumericArgumentError(f"{tool_name}.{name} must be a {expected}; booleans are not numeric arguments")

        if expected == "integer":
            if isinstance(value, int):
                number = value
            elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                number = int(value.strip())
            else:
                raise NumericArgumentError(f"{tool_name}.{name} must be an integer")
        else:
            if isinstance(value, (int, float)):
                number = float(value)
            elif isinstance(value, str):
                try:
                    number = float(value.strip())
                except ValueError as error:
                    raise NumericArgumentError(f"{tool_name}.{name} must be a number") from error
            else:
                raise NumericArgumentError(f"{tool_name}.{name} must be a number")
            if not math.isfinite(number):
                raise NumericArgumentError(f"{tool_name}.{name} must be a finite number")

        if "minimum" in schema and number < schema["minimum"]:
            raise NumericArgumentError(f"{tool_name}.{name} must be at least {schema['minimum']}")
        if "maximum" in schema and number > schema["maximum"]:
            raise NumericArgumentError(f"{tool_name}.{name} must be at most {schema['maximum']}")
        normalized[name] = number
    return normalized


def normalize_tool_arguments(tool_name, arguments):
    """Normalize numeric arguments against the shared Agentic tool schemas."""
    from backend.agent.planner import TOOL_DEFINITIONS

    definition = next(
        (
            tool["function"]
            for tool in TOOL_DEFINITIONS
            if tool.get("function", {}).get("name") == tool_name
        ),
        None,
    )
    if definition is None:
        return dict(arguments)
    properties = definition.get("parameters", {}).get("properties", {})
    return normalize_numeric_arguments(tool_name, arguments, properties)
