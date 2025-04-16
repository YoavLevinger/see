import sqlite3
import pandas as pd
import torch
import json
from sentence_transformers import SentenceTransformer, util

DB_SQL_PATH = "sdee_lite_description_vectorized.sql"
VECTOR_TABLE = "sbert_description_vectorized"
TOP_K = 5

def load_vectorized_data(sql_path):
    """Load vectorized embeddings from an existing SQL dump."""
    conn = sqlite3.connect(":memory:")
    with open(sql_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    df = pd.read_sql_query(f"SELECT * FROM {VECTOR_TABLE}", conn)
    # Join descriptions from repo_additional_info
    df_desc = pd.read_sql_query("SELECT owner, repo, description FROM repo_additional_info", conn)
    df = pd.merge(df, df_desc, on=["owner", "repo"], how="left")

    df = df.drop_duplicates(subset=["owner", "repo"])
    df["embedding"] = df["vectorized_description"].apply(
        lambda x: torch.tensor(json.loads(x), dtype=torch.float32)
    )
    return df

def get_top_k_similar_repos(description, top_k=TOP_K):
    """Encode new description and return top-k most similar repositories."""
    df = load_vectorized_data(DB_SQL_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model._target_device = "cpu"

    new_vec = torch.tensor(model.encode(description), dtype=torch.float32)
    df["similarity"] = df["embedding"].apply(lambda emb: float(util.cos_sim(new_vec, emb)))
    return df.sort_values(by="similarity", ascending=False).head(top_k)

# Example usage - marked-out for tests
# if __name__ == "__main__":
#     description_text = "A lightweight tool for deploying RESTful APIs in Python using Flask and SQLAlchemy."
#     df_vectorized = load_vectorized_data(DB_SQL_PATH)
#     top_k_repos = get_top_k_similar_repos(description_text, df_vectorized)
#
#     print("\n🔍 Top 10 Most Similar Projects:")
#     for _, row in top_k_repos.iterrows():
#         print(f"- {row['owner']}/{row['repo']} (Similarity: {row['similarity']:.4f})")
#         print(f"  📄 Description: {row['description']}\n")
