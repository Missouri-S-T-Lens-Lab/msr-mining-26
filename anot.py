import csv
import ollama

# Load long prompt text from file
def load_prompt(path="prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

CLASSIFICATION_PROMPT = load_prompt()


def classify_pr(pr, model="mistral"):
    """
    pr: dict with keys: agent, html_url, title, body, state
    returns: one-line category string
    """

    user_message = f"""
agent: {pr['agent']}
html_url: {pr['html_url']}
title: {pr['title']}
body: {pr['body']}
state: {pr['state']}
"""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": user_message}
        ],
        options={"temperature": 0}  # deterministic classifications
    )

    output = response["message"]["content"].strip()

    # You require: "Output a single line containing one category"
    category = output.splitlines()[0].strip()

    return category


def classify_tsv(input_path, output_path, model="mistral"):
    """
    Reads a TSV, classifies each row, writes a new TSV with category column.
    """

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames + ["category"]

        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for i, row in enumerate(reader, 1):
            print(f"Processing row {i}...", end="\r")
            category = classify_pr(row, model=model)
            row["category"] = category
            writer.writerow(row)

    print(f"\nDone. Output saved to {output_path}")


if __name__ == "__main__":
    inp = "Gold_100.tsv"
    op = "Anot_100_gpt.tsv"

    # model = "mistral-small"

    model = "gpt-oss:20b"

    classify_tsv(inp,op,model)