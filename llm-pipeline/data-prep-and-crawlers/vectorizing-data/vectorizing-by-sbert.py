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


# 🎯 Top K controls how many similar past projects we consider.
# value of 5 - 10 is chosen as a reasonable trade-off between stability (enough data points)
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




if __name__ == "__main__":
    print("🚀 Starting SBERT Similarity Description vectorizing...")

    if not os.path.exists(DB_SQL_PATH):
        print(f"🔧 Vectorizing dataset from original: {ORIGINAL_SQL}")
        conn = load_database(ORIGINAL_SQL)
        vectorize_descriptions(conn)
        save_database_to_sql(conn, DB_SQL_PATH)
    else:
        print(f"✅ Vectorized SQL already exists: {DB_SQL_PATH}")

