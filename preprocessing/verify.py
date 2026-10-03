import json

with open("C:\\Users\\harsh\\Desktop\\tigergraph_hackathon\\corpus\\corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Number of records:", len(data))

print("\nFirst record:")
print(json.dumps(data[0], indent=2, ensure_ascii=False))

print("\nFields:")
print(data[0].keys())