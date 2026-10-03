import json

input_file = "C:\\Users\\harsh\\Desktop\\tigergraph_hackathon\\corpus\\corpus.jsonl"
output_file = "C:\\Users\\harsh\\Desktop\\tigergraph_hackathon\\corpus\\corpus.json"

count = 0
with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line_number, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error on line {line_number}: {e}")
            raise
        # write back as ONE compact line, no indent
        fout.write(json.dumps(record, ensure_ascii=False))
        fout.write("\n")
        count += 1

print(f"Wrote {count} records to {output_file}")