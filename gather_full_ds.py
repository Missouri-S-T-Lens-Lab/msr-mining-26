# python gather_full_ds.py gold_full.tsv prs_with_comments.json
#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

# POP tables
PR_COMMENTS_PARQUET = "hf://datasets/hao-li/AIDev/pr_comments.parquet"
PR_REVIEWS_PARQUET = "hf://datasets/hao-li/AIDev/pr_reviews.parquet"
PR_REVIEW_COMMENTS_CANDIDATES = [
    "hf://datasets/hao-li/AIDev/pr_review_comments_v2.parquet",
    "hf://datasets/hao-li/AIDev/pr_review_comments.parquet",
]


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


def load_issue_comments(pr_ids_str: set[str]) -> dict[str, list[dict]]:
    """Map PR id (string) -> list of issue-style comments from pr_comments."""
    try:
        df = pd.read_parquet(PR_COMMENTS_PARQUET)
    except Exception as e:
        print(f"[!] Could not load pr_comments: {e}")
        return {}

    print(f"[+] pr_comments rows: {len(df)}")
    if "pr_id" not in df.columns:
        print("[!] pr_comments has no pr_id column")
        return {}

    dtypes = df.dtypes.to_dict()
    df["pr_id_str"] = df["pr_id"].astype(str)

    df = df[df["pr_id_str"].isin(pr_ids_str)]
    print(f"[+] pr_comments matching TSV PRs: {len(df)} rows")

    by_pr: dict[str, list[dict]] = {}
    for pr_id_str, group in df.groupby("pr_id_str"):
        by_pr[pr_id_str] = [
            normalize_row(row.drop(labels=["pr_id_str"]), dtypes)
            for _, row in group.iterrows()
        ]
    return by_pr


def load_reviews(pr_ids_str: set[str]):
    """
    Load pr_reviews and return:
      - reviews_by_pr: PR id string -> list[review dict]
      - review_id_to_pr_str: review id string -> PR id string
    """
    try:
        df = pd.read_parquet(PR_REVIEWS_PARQUET)
    except Exception as e:
        print(f"[!] Could not load pr_reviews: {e}")
        return {}, {}

    print(f"[+] pr_reviews rows: {len(df)}")
    if "pr_id" not in df.columns or "id" not in df.columns:
        print("[!] pr_reviews must have pr_id and id columns")
        return {}, {}

    dtypes = df.dtypes.to_dict()
    df["pr_id_str"] = df["pr_id"].astype(str)
    df["review_id_str"] = df["id"].astype(str)

    df = df[df["pr_id_str"].isin(pr_ids_str)]
    print(f"[+] pr_reviews matching TSV PRs: {len(df)} rows")

    reviews_by_pr: dict[str, list[dict]] = {}
    review_id_to_pr_str: dict[str, str] = {}

    for pr_id_str, group in df.groupby("pr_id_str"):
        reviews = []
        for _, row in group.iterrows():
            row_dict = normalize_row(row.drop(labels=["pr_id_str", "review_id_str"]), dtypes)
            reviews.append(row_dict)
            review_id_to_pr_str[row["review_id_str"]] = pr_id_str
        reviews_by_pr[pr_id_str] = reviews

    return reviews_by_pr, review_id_to_pr_str


def load_review_comments(review_id_to_pr_str: dict[str, str]) -> dict[str, list[dict]]:
    """
    Load pr_review_comments(_v2) and attach each comment to PR via:
      pr_review_comments.pull_request_review_id == pr_reviews.id
    """
    df = None
    last_err = None
    for path in PR_REVIEW_COMMENTS_CANDIDATES:
        try:
            df = pd.read_parquet(path)
            print(f"[+] Loaded pr_review_comments from {path}: {len(df)} rows")
            break
        except Exception as e:
            last_err = e
            continue

    if df is None:
        print(f"[!] Could not load any pr_review_comments table: {last_err}")
        return {}

    if "pull_request_review_id" not in df.columns:
        print("[!] pr_review_comments has no pull_request_review_id column")
        return {}

    dtypes = df.dtypes.to_dict()
    df["review_id_str"] = df["pull_request_review_id"].astype(str)

    valid_review_ids = set(review_id_to_pr_str.keys())
    df = df[df["review_id_str"].isin(valid_review_ids)]
    print(f"[+] pr_review_comments matching known reviews: {len(df)} rows")

    out: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        review_id_str = row["review_id_str"]
        pr_id_str = review_id_to_pr_str.get(review_id_str)
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

    # Load your POP PR TSV
    pr_df = pd.read_csv(tsv, sep="\t")
    print(f"[+] Loaded TSV: {len(pr_df)} rows")
    if "id" not in pr_df.columns:
        raise ValueError("TSV must contain column 'id' (pull_request.id)")

    pr_dtypes = pr_df.dtypes.to_dict()

    # Use string IDs for joining
    pr_df["id_str"] = pr_df["id"].astype(str)
    pr_ids_str: set[str] = set(pr_df["id_str"].tolist())
    print(f"[+] Unique TSV PR ids: {len(pr_ids_str)}")

    # 1) Issue comments: pr_comments.pr_id = pull_request.id
    issue_comments_by_pr = load_issue_comments(pr_ids_str)

    # 2) Reviews: pr_reviews.pr_id = pull_request.id
    reviews_by_pr, review_id_to_pr = load_reviews(pr_ids_str)

    # 3) Review comments: pr_review_comments.pull_request_review_id = pr_reviews.id
    review_comments_by_pr = load_review_comments(review_id_to_pr)

    # 4) Build output: TSV columns (typed) + comments
    records = []
    for _, row in pr_df.iterrows():
        pr_id_str = row["id_str"]

        base = normalize_row(row.drop(labels=["id_str"]), pr_dtypes)
        base["issue_comments"] = issue_comments_by_pr.get(pr_id_str, [])
        base["review_comments"] = review_comments_by_pr.get(pr_id_str, [])
        # if you also want the reviews themselves:
        # base["reviews"] = reviews_by_pr.get(pr_id_str, [])

        records.append(base)

    out_path = Path(output_json)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved joined PR + comments JSON to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Join POP PR TSV (id = pull_request.id) with pr_comments + pr_review_comments via pr_reviews."
    )
    parser.add_argument("pr_tsv", help="Path to POP PR TSV (tab-separated, must contain column 'id')")
    parser.add_argument("output_json", help="Output JSON file path")

    args = parser.parse_args()
    main(args.pr_tsv, args.output_json)
