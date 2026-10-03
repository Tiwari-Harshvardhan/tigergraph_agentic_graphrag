import argparse
import json
import sys
from pathlib import Path

from evaluation.metrics import evaluate_records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark JSONL results.")
    parser.add_argument("--input", required=True, help="Benchmark result JSONL")
    parser.add_argument("--output", help="Evaluation JSON output path")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    try:
        with input_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
        report = evaluate_records(records)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".evaluation.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output_path), "pipelines": report["by_pipeline"]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"success": False, "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())