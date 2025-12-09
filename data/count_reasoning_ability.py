import argparse
import json
import re


TOTAL_COUNT_PATTERN = re.compile(r'"total_count"\s*:\s*(\d+)', re.IGNORECASE)


def extract_total_count(field_text: str | None) -> int:
    """Return the integer total_count embedded in a reasoning field."""
    if field_text is None:
        return 0

    text = field_text.strip()
    if not text or text.upper().startswith("ERROR"):
        return 0

    match = TOTAL_COUNT_PATTERN.search(text)
    if not match:
        return 0
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(
        description="Count how many samples contain deduction / induction / abduction outputs."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to *_reasoning_analysis.json file",
    )
    args = parser.parse_args()

    with open(args.input_path, "r") as f:
        data = json.load(f)

    total = len(data)
    ded = ind = abd = 0

    for item in data:
        analysis = item.get("reasoning_analysis", {})
        if extract_total_count(analysis.get("deduction")) > 0:
            ded += 1
        if extract_total_count(analysis.get("induction")) > 0:
            ind += 1
        if extract_total_count(analysis.get("abduction")) > 0:
            abd += 1

    def pct(x):
        return 0 if total == 0 else 100 * x / total

    print(f"Total samples: {total}")
    print(f"Deduction: {ded} ({pct(ded):.2f}%)")
    print(f"Induction: {ind} ({pct(ind):.2f}%)")
    print(f"Abduction: {abd} ({pct(abd):.2f}%)")

if __name__ == "__main__":
    main()