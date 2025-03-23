import os
import sqlite3
import pandas as pd
import numpy as np
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Disable GPU to avoid NCCL backend error
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# === CONFIG ===
SQL_PATH = "sdee_lite_description_vectorized.sql"
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

    # Merge and deduplicate
    df = df_vec.merge(df_effort, on=["owner", "repo"]).merge(df_desc, on=["owner", "repo"])
    print(f"📊 Total records before deduplication: {len(df)}")
    df = df.drop_duplicates(subset=["owner", "repo"])
    print(f"📊 Total records after deduplication: {len(df)}")

    # Exclude near-zero effort values
    df = df[df["actual_effort"] > 1e-4]
    print(f"📊 Total records after filtering near-zero efforts: {len(df)}")

    # Convert vector from string to tensor
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

        # Get top-k most similar
        top_matches = comparison_df.sort_values(by="similarity", ascending=False).head(top_k)
        weights = top_matches["similarity"].values
        est_effort = np.average(top_matches["actual_effort"], weights=weights)

        # Accuracy
        absolute_error = abs(est_effort - target_effort)
        percentage_error = absolute_error / target_effort
        accuracy = 1.0 - percentage_error

        results.append({
            "owner": row["owner"],
            "repo": row["repo"],
            "description": row["description"],
            "actual_effort": target_effort,
            "sbert_estimated_effort": est_effort,
            "accuracy": round(accuracy, 4),
            "accuracy_percent": round(accuracy * 100, 2)
        })

    return pd.DataFrame(results)


def save_results_to_db(conn, result_df):
    """Save final result to the database."""
    result_df.to_sql(RESULT_TABLE, conn, index=False, if_exists="replace")
    print(f"✅ Results saved to table: {RESULT_TABLE}")


def calculate_metrics(df):
    """Calculate regression validation metrics."""
    y_true = df["actual_effort"].values
    y_pred = df["sbert_estimated_effort"].values

    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n📈 Validation Metrics:")
    print(f"MAE  (Mean Absolute Error):        {mae:.2f}")
    print(f"MAPE (Mean Absolute % Error):      {mape:.2f}%")
    print(f"RMSE (Root Mean Squared Error):    {rmse:.2f}")
    print(f"R²    (Coefficient of Determination): {r2:.4f}")


def main():
    print("🚀 Loading data and preparing embeddings...")
    conn, df = load_data()

    print("🤖 Running SBERT similarity estimation (leave-one-out)...")
    result_df = estimate_efforts(df)

    print("💾 Saving results to in-memory database...")
    save_results_to_db(conn, result_df)

    print("📊 Top 5 Sample Results:")
    print(result_df.head(5))

    calculate_metrics(result_df)

    # Optional CSV export
    result_df.to_csv("sbert_effort_estimation_results.csv", index=False)
    print("📝 Also saved to: sbert_effort_estimation_results.csv")


if __name__ == "__main__":
    main()
