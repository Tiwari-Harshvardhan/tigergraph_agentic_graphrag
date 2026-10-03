import re
import unicodedata
from decimal import Decimal, InvalidOperation


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}
_ARTICLES = {"a", "an", "the"}


def normalize_answer(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\bq\d+(?:_c\d+)?\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(token for token in text.split() if token not in _ARTICLES)


def _parse_number_words(text: str):
    words = re.findall(r"[a-z]+", normalize_answer(text))
    number_words = set(_NUMBER_WORDS) | {"and", "hundred", "thousand", "million", "billion"}
    runs = []
    current_run = []
    for word in words + [""]:
        if word in number_words:
            current_run.append(word)
        elif current_run:
            runs.append(current_run)
            current_run = []

    values = []
    for run in runs:
        total = 0
        current = 0
        for word in run:
            if word == "and":
                continue
            if word in _NUMBER_WORDS:
                current += _NUMBER_WORDS[word]
            elif word == "hundred":
                current = max(current, 1) * 100
            else:
                scale = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}[word]
                total += max(current, 1) * scale
                current = 0
        values.append(Decimal(total + current))
    return values


def _numeric_values(text: str):
    cleaned = re.sub(r"\[[^\]]*\]", " ", text)
    cleaned = re.sub(r"\bq\d+(?:_c\d+)?\b", " ", cleaned, flags=re.IGNORECASE)
    raw_values = re.findall(r"(?<![a-z])[-+]?\d[\d,]*(?:\.\d+)?", cleaned, flags=re.IGNORECASE)
    values = []
    for raw in raw_values:
        try:
            values.append(Decimal(raw.replace(",", "")))
        except InvalidOperation:
            continue
    # Answers often mix a written answer ("five events") with unrelated
    # numeric context ("73 competitors"). Capture both representations.
    values.extend(_parse_number_words(cleaned))
    return values


def _list_values(value):
    if isinstance(value, (list, tuple, set)):
        return [normalize_answer(item) for item in value]
    text = str(value)
    if "," in text or ";" in text or "\n" in text or " and " in text.casefold():
        parts = re.split(r"[,;\n]+|\s+and\s+", text, flags=re.IGNORECASE)
        return [normalize_answer(part) for part in parts if normalize_answer(part)]
    return None


def evaluate_answer(generated, expected, question_type: str | None = None) -> dict:
    if expected is None:
        return {"correct": None, "method": "unavailable", "normalized_generated": normalize_answer(generated or "")}
    if generated is None or not str(generated).strip():
        return {"correct": False, "method": "missing_answer", "normalized_generated": ""}

    expected_values = list(expected) if isinstance(expected, (list, tuple, set)) else [expected]
    normalized_generated = normalize_answer(generated)
    normalized_expected = [normalize_answer(value) for value in expected_values]
    if normalized_generated in normalized_expected:
        return {"correct": True, "method": "normalized_exact", "normalized_generated": normalized_generated}

    if len(expected_values) == 1:
        expected_text = str(expected_values[0])
        expected_numbers = _numeric_values(expected_text)
        actual_numbers = _numeric_values(str(generated))
        expected_is_numeric = normalize_answer(expected_text).replace(" ", "").replace(",", "").replace(".", "").isdigit()
        if expected_is_numeric and len(expected_numbers) == 1 and expected_numbers[0] in actual_numbers:
            return {"correct": True, "method": "numeric", "normalized_generated": normalized_generated}

        if (question_type or "").casefold() == "superlative":
            expected_norm = normalized_expected[0]
            generated_tokens = normalized_generated.split()
            if len(generated_tokens) >= 2:
                phrase = " ".join(generated_tokens)
                if expected_norm.count(phrase) == 1:
                    return {
                        "correct": True,
                        "method": "unique_superlative_phrase",
                        "normalized_generated": normalized_generated,
                    }

    if len(expected_values) > 1:
        generated_list = _list_values(generated)
        if generated_list is not None and set(generated_list) == set(normalized_expected):
            return {"correct": True, "method": "list_set", "normalized_generated": normalized_generated}

    return {"correct": False, "method": "no_match", "normalized_generated": normalized_generated}
