import os
import sqlite3
import pandas as pd
import numpy as np
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Disable GPU to avoid NCCL backend error
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# === CONFIG ===
SQL_PATH = "sdee_lite_description_vectorized.sql"  # <- path to your vectorized SQL dump
TOP_K = 5
RESULT_TABLE = "sbert_effort_estimation_results"


def load_data():
    """Load vectorized SQL database into memory and return merged DataFrame."""
    conn = sqlite3.connect(":memory:")
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    df_vec = pd.read_sql_query("SELECT * FROM sbert_description_vectorized", conn)
    df_effort = pd.read_sql_query("SELECT owner, repo, effort as actual_effort FROM avg_repo_effort", conn)
    df_desc = pd.read_sql_query("SELECT owner, repo, description FROM repo_additional_info", conn)

    # Merge and clean
    df = df_vec.merge(df_effort, on=["owner", "repo"]).merge(df_desc, on=["owner", "repo"])
    df = df.drop_duplicates(subset=["owner", "repo"])
    # Exclude records with effort ≈ 0, which are likely invalid or noise
    # df = df[df["actual_effort"] > 1e-4]

    df["embedding"] = df["vectorized_description"].apply(lambda x: torch.tensor(json.loads(x), dtype=torch.float32))
    return conn, df


def estimate_efforts(df, top_k=TOP_K):
    """Estimate efforts using leave-one-out similarity."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Estimating efforts"):
        target_vec = row["embedding"]
        target_effort = row["actual_effort"]

        # Leave-one-out
        comparison_df = df.drop(index=idx).reset_index(drop=True)
        comparison_df["similarity"] = comparison_df["embedding"].apply(
            lambda emb: float(util.cos_sim(target_vec, emb))
        )

        top_matches = comparison_df.sort_values(by="similarity", ascending=False).head(top_k)
        weights = top_matches["similarity"].values
        est_effort = np.average(top_matches["actual_effort"], weights=weights)

        accuracy = 1.0 - abs(est_effort - target_effort) / target_effort

        results.append({
            "owner": row["owner"],
            "repo": row["repo"],
            "actual_effort": target_effort,
            "sbert_estimated_effort": est_effort,
            "accuracy": round(accuracy, 4)
        })

    return pd.DataFrame(results)


def save_results_to_db(conn, result_df):
    """Save final result to the database."""
    result_df.to_sql(RESULT_TABLE, conn, index=False, if_exists="replace")
    print(f"✅ Results saved to table: {RESULT_TABLE}")


def main():
    print("🚀 Loading data and preparing embeddings...")
    conn, df = load_data()

    print("🤖 Running SBERT similarity estimation (leave-one-out)...")
    result_df = estimate_efforts(df)

    print("💾 Saving results to in-memory database...")
    save_results_to_db(conn, result_df)

    print("📊 Top 5 Sample Results:")
    print(result_df.head(5))

    # Optionally export to CSV
    result_df.to_csv("sbert_effort_estimation_results.csv", index=False)
    print("📝 Also saved to: sbert_effort_estimation_results.csv")


if __name__ == "__main__":
    main()
