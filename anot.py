#!/usr/bin/env python3
import json
import csv
from pathlib import Path

import ollama


# ----------------------- CONFIG -----------------------

PROMPT_PATH = "few-shot-prompt.txt"   # classification instructions
DEFAULT_MODEL = "gpt-oss:20b"  # or "mistral", "mistral-small", etc.
MAX_COMMENTS = 5
MAX_COMMENT_CHARS = 300


# ----------------------- PROMPT LOADING -----------------------

def load_prompt(path=PROMPT_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


CLASSIFICATION_PROMPT = load_prompt()


# ----------------------- COMMENT FORMATTING -----------------------

def format_comments(comments, label):
    """
    comments: list[dict]
    label: 'Issue' or 'Review'
    Returns: a readable, truncated multi-line string.
    """
    if not comments:
        return f"{label} comments: (none)"

    lines = [f"{label} comments (showing up to {MAX_COMMENTS}):"]
    for i, c in enumerate(comments[:MAX_COMMENTS], 1):
        user = (
            c.get("user_login")
            or c.get("user")
            or c.get("author_login")
            or ""
        )
        body = str(c.get("body", "") or "")
        body = body.replace("\n", " ").strip()
        if len(body) > MAX_COMMENT_CHARS:
            body = body[:MAX_COMMENT_CHARS] + "…"

        if user:
            lines.append(f"- [{i}] {user}: {body}")
        else:
            lines.append(f"- [{i}] {body}")

    if len(comments) > MAX_COMMENTS:
        lines.append(f"... ({len(comments) - MAX_COMMENTS} more not shown)")

    return "\n".join(lines)


# ----------------------- LLM CALL -----------------------

def classify_pr(pr, model=DEFAULT_MODEL):
    """
    pr dict schema (your JSON):
      id, title, body, state,
      num_success_checks, num_failed_checks, num_neutral_checks,
      overall_ci_status,
      issue_comments: list[dict],
      review_comments: list[dict]

    returns: category string (first line of model output)
    """

    issue_comments = pr.get("issue_comments", []) or []
    review_comments = pr.get("review_comments", []) or []

    issue_comments_block = format_comments(issue_comments, "Issue")
    review_comments_block = format_comments(review_comments, "Review")

    user_message = f""" Classify the following PR. Informations :
    
id: {pr.get('id', '')}
title: {pr.get('title', '')}
body:
{pr.get('body', '')}

state: {pr.get('state', '')}

num_success_checks: {pr.get('num_success_checks', '')}
num_failed_checks: {pr.get('num_failed_checks', '')}
num_neutral_checks: {pr.get('num_neutral_checks', '')}
overall_ci_status: {pr.get('overall_ci_status', '')}

{issue_comments_block}

{review_comments_block}
""".strip()

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0},  # deterministic
    )

    output = response["message"]["content"].strip()
    category = output.splitlines()[0].strip()
    return category


# ----------------------- JSON I/O -----------------------

def load_json_records(path):
    """
    Load PRs from either:
      - JSON array file
      - or JSONL file (one JSON per line)
    """
    p = Path(path)

    # Try JSON array first
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# ----------------------- TSV OUTPUT -----------------------

def write_tsv(records, path):
    """
    Write TSV with your fields + 'category'.
    issue_comments and review_comments are JSON-stringified.
    """
    if not records:
        with open(path, "w", encoding="utf-8", newline="") as f:
            pass
        return

    # Base fields
    base_fields = [
        "id",
        "title",
        "body",
        "state",
        "num_success_checks",
        "num_failed_checks",
        "num_neutral_checks",
        "overall_ci_status",
        "issue_comments",
        "review_comments",
    ]
    fieldnames = base_fields + ["category"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for r in records:
            row = {}
            for col in base_fields:
                v = r.get(col, "")

                if col in ("issue_comments", "review_comments"):
                    # stringify arrays as JSON
                    row[col] = json.dumps(v, ensure_ascii=False)
                else:
                    row[col] = v

            row["category"] = r.get("category", "")
            writer.writerow(row)


# ----------------------- MAIN PIPELINE -----------------------

def classify_json(input_path, output_json_path, output_tsv_path, model=DEFAULT_MODEL):
    prs = load_json_records(input_path)
    print(f"[+] Loaded {len(prs)} PR records from {input_path}")

    for i, pr in enumerate(prs, 1):
        print(f"Classifying PR {i}/{len(prs)}...", end="\r")
        category = classify_pr(pr, model=model)
        pr["category"] = category

    print("\n[+] Classification complete. Writing outputs...")

    write_json(prs, output_json_path)
    write_tsv(prs, output_tsv_path)

    print(f"[✓] JSON saved to: {output_json_path}")
    print(f"[✓] TSV saved to:  {output_tsv_path}")


# ----------------------- ENTRY POINT -----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify PRs from JSON using Ollama and write JSON + TSV outputs."
    )
    parser.add_argument("input_json", help="Input JSON (array or JSONL)")
    parser.add_argument("--out-json", default="prs_labeled.json", help="Output JSON file")
    parser.add_argument("--out-tsv", default="prs_labeled.tsv", help="Output TSV file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")

    args = parser.parse_args()

    classify_json(args.input_json, args.out_json, args.out_tsv, model=args.model)




