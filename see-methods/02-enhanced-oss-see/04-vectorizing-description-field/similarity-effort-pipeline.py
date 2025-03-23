import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import json
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

VECTOR_TABLE = "sbert_description_vectorized"
DB_SQL_PATH = "sdee_lite_description_vectorized.sql"
ORIGINAL_SQL = "sdee_lite_description_cleaned.sql"
# VECTOR_TABLE = "sbert_description_vectorized"
# SQL_PATH = "sdee_lite_description_vectorized.sql"


# 🎯 Top K controls how many similar past projects we consider.
# 5 is chosen as a reasonable trade-off between stability (enough data points)
# and relevance (only most similar matches considered).
TOP_K = 10

def load_database(sql_path):
    """Load SQL dump into memory"""
    conn = sqlite3.connect(":memory:")
    with open(sql_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    return conn


def vectorize_descriptions(conn, model_name="all-MiniLM-L6-v2"):
    """Vectorize repo descriptions and save them to a new table in DB"""
    print("📦 Loading SBERT model...")
    model = SentenceTransformer(model_name)

    df_desc = pd.read_sql_query("SELECT owner, repo, description FROM repo_additional_info", conn)
    print(f"🧠 Vectorizing {len(df_desc)} descriptions...")
    df_desc["vectorized_description"] = df_desc["description"].apply(lambda x: model.encode(x).tolist())

    # Save to new table
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {VECTOR_TABLE}")
    cursor.execute(f"""
        CREATE TABLE {VECTOR_TABLE} (
            owner TEXT,
            repo TEXT,
            vectorized_description TEXT
        )
    """)
    for _, row in df_desc.iterrows():
        cursor.execute(
            f"INSERT INTO {VECTOR_TABLE} (owner, repo, vectorized_description) VALUES (?, ?, ?)",
            (row["owner"], row["repo"], json.dumps(row["vectorized_description"]))
        )
    conn.commit()
    print(f"✅ Vectorized data saved to '{VECTOR_TABLE}'")


def save_database_to_sql(conn, output_path):
    """Dump the in-memory database to a SQL file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for line in conn.iterdump():
            f.write(f"{line}\n")
    print(f"💾 Updated database saved to '{output_path}'")




def load_data(conn):
    """Load vectorized descriptions and effort values."""
    df_vec = pd.read_sql_query(f"SELECT * FROM {VECTOR_TABLE}", conn)
    # df_effort = pd.read_sql_query("SELECT owner, repo, effort FROM avg_repo_effort", conn)
    df_effort = pd.read_sql_query("SELECT owner, repo, effort_score AS effort FROM repo_additional_info", conn)
    df = pd.merge(df_vec, df_effort, on=["owner", "repo"])

    # ✅ REMOVE DUPLICATES that result from dirty source data
    df = df.drop_duplicates(subset=["owner", "repo"])

    df = df[df["effort"] > 1e-4]  # filter out near-zero values
    return df


def encode_new_description(model, description):
    """Encode the new description using SBERT."""
    return torch.tensor(model.encode(description), dtype=torch.float32)


def restore_embeddings(df):
    """Convert JSON vectors into float32 tensors."""
    df["embedding"] = df["vectorized_description"].apply(
        lambda x: torch.tensor(json.loads(x), dtype=torch.float32)
    )
    return df


def calculate_similarity(df, new_vec):
    """Compute cosine similarity to all embeddings."""
    df["similarity"] = df["embedding"].apply(lambda emb: float(util.cos_sim(new_vec, emb)))
    return df.sort_values(by="similarity", ascending=False)


def summarize_top_matches(df, top_k=TOP_K):
    """Select and summarize top-k matches with normalized weights."""
    top = df.head(top_k).copy()
    weights = top["similarity"].values
    top["normalized_weight"] = weights / weights.sum()
    predicted_effort = np.average(top["effort"], weights=weights)
    return top, predicted_effort


def plot_top_matches(df):
    """Visualize similarity and effort of top-k matches."""
    repos = df["repo"]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(repos, df["similarity"], color='skyblue')
    ax2.plot(repos, df["effort"], color='orange', marker='o')

    ax1.set_ylabel("Cosine Similarity")
    ax2.set_ylabel("Effort (person-months)")
    ax1.set_xlabel("Top Similar Projects")
    ax1.set_title("Top Similar Projects and Their Efforts")

    fig.tight_layout()
    plt.show()


def predict_effort_similarity(sql_path, description):
    print("🔄 Loading database...")
    conn = load_database(sql_path)
    df = load_data(conn)

    print("📦 Loading SBERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model._target_device = "cpu"  # Optional: ensure consistent device

    print("🧠 Encoding description and restoring embeddings...")
    new_vec = encode_new_description(model, description)
    df = restore_embeddings(df)

    print("📊 Calculating similarity...")
    df_similar = calculate_similarity(df, new_vec)
    top_matches, predicted_effort = summarize_top_matches(df_similar)

    # Fetch descriptions from repo_additional_info
    df_desc = pd.read_sql_query("SELECT owner, repo, description FROM repo_additional_info", conn)
    df_desc = df_desc.drop_duplicates(subset=["owner", "repo"])

    top_matches = pd.merge(top_matches, df_desc, on=["owner", "repo"], how="left")
    top_matches = top_matches.drop_duplicates(subset=["owner", "repo"])

    print("\n📌 Top Similar Projects:")
    print(f"{'Owner':<20} {'Repo':<35} {'Similarity':>10} {'Effort':>10} {'Weight':>10}")
    print("-" * 90)
    for _, row in top_matches.iterrows():
        print(
            f"{row['owner']:<20} {row['repo']:<35} {row['similarity']:.4f} {row['effort']:10.2f} {row['normalized_weight']:10.3f}")
        # print(f"    📄 Description: {row['description'][:200]}...\n")  # truncate long descriptions
        print(f"    📄 Description: {row['description']}\n")

    print(f"\n🎯 Predicted Effort: {predicted_effort:.2f} person-months")

    # plot_top_matches(top_matches)

    return predicted_effort

if __name__ == "__main__":
    print("🚀 Starting SBERT Similarity Effort Estimation Pipeline...")

    if not os.path.exists(DB_SQL_PATH):
        print(f"🔧 Vectorizing dataset from original: {ORIGINAL_SQL}")
        conn = load_database(ORIGINAL_SQL)
        vectorize_descriptions(conn)
        save_database_to_sql(conn, DB_SQL_PATH)
    else:
        print(f"✅ Vectorized SQL already exists: {DB_SQL_PATH}")

    # Run prediction for an example
    example_desc = "A lightweight tool for deploying RESTful APIs in Python using Flask and SQLAlchemy."
    predict_effort_similarity(DB_SQL_PATH, example_desc)
