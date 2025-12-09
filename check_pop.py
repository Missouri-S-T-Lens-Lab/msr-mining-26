#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

# POP subset tables
POP_PULL_REQUEST = "hf://datasets/hao-li/AIDev/pull_request.parquet"
PR_COMMENTS = "hf://datasets/hao-li/AIDev/pr_comments.parquet"
PR_REVIEWS = "hf://datasets/hao-li/AIDev/pr_reviews.parquet"
PR_REVIEW_COMMENTS_V2 = "hf://datasets/hao-li/AIDev/pr_review_comments_v2.parquet"


def normalize_row(row: pd.Series, dtypes: dict):
    """
    Per-column typing:
      - object/text: NaN -> "", value -> str(...)
      - non-object (numeric/bool/datetime): NaN -> None, keep native type
    """
    out = {}
    for col, val in row.items():
        dtype = dtypes[col]
        if dtype == "object":
            out[col] = "" if pd.isna(val) else str(val)
        else:
            out[col] = None if pd.isna(val) else val
    return out


def load_pop_ids():
    """Load POP pull_request table and return its IDs as strings."""
    df = pd.read_parquet(POP_PULL_REQUEST)
    if "id" not in df.columns:
        raise RuntimeError("POP pull_request.parquet has no 'id' column")
    df["id_str"] = df["id"].astype(str)
    pop_ids = set(df["id_str"].tolist())
    print(f"[+] POP pull_request: {len(df)} rows, {len(pop_ids)} unique ids")
    return pop_ids


def load_issue_comments(pop_ids_str: set[str]) -> dict[str, list[dict]]:
    """pr_comments.pr_id -> issue_comments, restricted to POP ids."""
    try:
        df = pd.read_parquet(PR_COMMENTS)
    except Exception as e:
        print(f"[!] Could not load pr_comments: {e}")
        return {}

    print(f"[+] pr_comments rows: {len(df)}")
    if "pr_id" not in df.columns:
        print("[!] pr_comments has no pr_id column")
        return {}

    dtypes = df.dtypes.to_dict()

    df["pr_id_str"] = df["pr_id"].astype(str)
    df = df[df["pr_id_str"].isin(pop_ids_str)]
    print(f"[+] pr_comments matching POP PRs: {len(df)} rows")

    by_pr: dict[str, list[dict]] = {}
    for pr_id_str, group in df.groupby("pr_id_str"):
        by_pr[pr_id_str] = [
            normalize_row(row.drop(labels=["pr_id_str"]), dtypes)
            for _, row in group.iterrows()
        ]
    return by_pr


def load_reviews(pop_ids_str: set[str]):
    """
    pr_reviews.pr_id -> reviews, and id -> pr_id mapping
    """
    try:
        df = pd.read_parquet(PR_REVIEWS)
    except Exception as e:
        print(f"[!] Could not load pr_reviews: {e}")
        return {}, {}

    print(f"[+] pr_reviews rows: {len(df)}")
    if "pr_id" not in df.columns or "id" not in df.columns:
        print("[!] pr_reviews must contain pr_id and id")
        return {}, {}

    dtypes = df.dtypes.to_dict()

    df["pr_id_str"] = df["pr_id"].astype(str)
    df["review_id_str"] = df["id"].astype(str)

    df = df[df["pr_id_str"].isin(pop_ids_str)]
    print(f"[+] pr_reviews matching POP PRs: {len(df)} rows")

    reviews_by_pr: dict[str, list[dict]] = {}
    review_id_to_pr: dict[str, str] = {}

    for pr_id_str, group in df.groupby("pr_id_str"):
        reviews = []
        for _, row in group.iterrows():
            row_dict = normalize_row(row.drop(labels=["pr_id_str", "review_id_str"]), dtypes)
            reviews.append(row_dict)
            review_id_to_pr[row["review_id_str"]] = pr_id_str
        reviews_by_pr[pr_id_str] = reviews

    return reviews_by_pr, review_id_to_pr


def load_review_comments_v2(review_id_to_pr: dict[str, str]) -> dict[str, list[dict]]:
    """pr_review_comments_v2.pull_request_review_id -> pr_reviews.id -> PR."""
    try:
        df = pd.read_parquet(PR_REVIEW_COMMENTS_V2)
    except Exception as e:
        print(f"[!] Could not load pr_review_comments_v2: {e}")
        return {}

    print(f"[+] pr_review_comments_v2 rows: {len(df)}")
    if "pull_request_review_id" not in df.columns:
        print("[!] pr_review_comments_v2 has no pull_request_review_id column")
        return {}

    dtypes = df.dtypes.to_dict()

    df["review_id_str"] = df["pull_request_review_id"].astype(str)
    valid_review_ids = set(review_id_to_pr.keys())
    df = df[df["review_id_str"].isin(valid_review_ids)]
    print(f"[+] pr_review_comments_v2 matching known reviews: {len(df)} rows")

    out: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        review_id_str = row["review_id_str"]
        pr_id_str = review_id_to_pr.get(review_id_str)
        if pr_id_str is None:
            continue
        if pr_id_str not in out:
            out[pr_id_str] = []
        out[pr_id_str].append(
            normalize_row(row.drop(labels=["review_id_str"]), dtypes)
        )

    return out


def main(tsv_path: str, output_json: str):
    tsv = Path(tsv_path)
    if not tsv.is_file():
        raise FileNotFoundError(f"TSV not found: {tsv}")

    # Your TSV (likely from all_pull_request viewer)
    pr_df = pd.read_csv(tsv, sep="\t")
    print(f"[+] Loaded TSV: {len(pr_df)} rows")
    if "id" not in pr_df.columns:
        raise ValueError("TSV must contain column 'id'")

    pr_dtypes = pr_df.dtypes.to_dict()
    pr_df["id_str"] = pr_df["id"].astype(str)
    tsv_ids_str = set(pr_df["id_str"].tolist())
    print(f"[+] Unique TSV PR ids: {len(tsv_ids_str)}")

    # POP pull_request IDs
    pop_ids_str = load_pop_ids()

    # Intersection = PRs that have POP metadata & comment tables behind them
    pop_overlap = tsv_ids_str & pop_ids_str
    print(f"[+] TSV PRs in POP subset: {len(pop_overlap)}")

    # Load comments/reviews only for POP PRs
    issue_comments_by_pr = load_issue_comments(pop_overlap)
    reviews_by_pr, review_id_to_pr = load_reviews(pop_overlap)
    review_comments_by_pr = load_review_comments_v2(review_id_to_pr)

    # Build final JSON
    records = []
    for _, row in pr_df.iterrows():
        pr_id_str = row["id_str"]
        in_pop = pr_id_str in pop_overlap

        base = normalize_row(row.drop(labels=["id_str"]), pr_dtypes)
        base["in_pop"] = in_pop
        base["issue_comments"] = issue_comments_by_pr.get(pr_id_str, [])
        base["review_comments"] = review_comments_by_pr.get(pr_id_str, [])
        # optional:
        # base["reviews"] = reviews_by_pr.get(pr_id_str, []) if in_pop else []

        records.append(base)

    out_path = Path(output_json)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved joined PR + comments JSON to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Join PRs from all_pull_request TSV with POP comment tables (pr_comments, pr_reviews, pr_review_comments_v2)."
    )
    parser.add_argument("pr_tsv", help="Path to TSV exported from AIDev viewer (must contain column 'id')")
    parser.add_argument("output_json", help="Output JSON path")

    args = parser.parse_args()
    main(args.pr_tsv, args.output_json)
