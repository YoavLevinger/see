import sqlite3
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import uvicorn

# Load SBERT model globally to avoid reloading
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 🛠️ DATABASE OPERATIONS (IN-MEMORY)
# -------------------------------

def load_data(sql_file="../0-artifacts/sdee_mysql_description_cleaned.sql"):
    """Loads the database into memory from an SQL file."""
    print(f"📂 Loading data from {sql_file} into memory...")

    # Connect to in-memory SQLite
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    try:
        with open(sql_file, "r", encoding="utf-8") as file:
            sql_script = file.read()
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decoding failed. Using latin-1 encoding as fallback.")
        with open(sql_file, "r", encoding="latin-1") as file:
            sql_script = file.read()

    cursor.executescript(sql_script)

    # Load the 02-dataset into a Pandas DataFrame
    query = "SELECT * FROM repo_additional_info;"
    df = pd.read_sql_query(query, conn)

    print("✅ Data loaded into memory successfully!")
    return df, conn


# -------------------------------
# 🔥 SBERT VECTORIZATION (FIXED)
# -------------------------------

def vectorize_descriptions(df, sql_save_path="sdee_lite_desc_vect.sql"):
    """Uses SBERT to convert descriptions to vectors and saves them."""
    print("🔥 Vectorizing descriptions using SBERT...")

    # Handle missing descriptions (fill NaN with a placeholder text)
    df["description"] = df["description"].fillna("No description provided")

    # Convert descriptions to embeddings
    df["embedding"] = df["description"].apply(lambda x: sbert_model.encode(str(x)))

    # Save vectorized data to an SQL script
    conn = sqlite3.connect(sql_save_path)
    df.to_sql("repo_additional_info_vectorized", conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Vectorized data saved to {sql_save_path}")

    # Load vectorized data into memory
    conn_mem = sqlite3.connect(":memory:")
    cursor = conn_mem.cursor()

    try:
        with open(sql_save_path, "r", encoding="utf-8") as file:
            sql_script = file.read()
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decoding failed. Using latin-1 encoding as fallback.")
        with open(sql_save_path, "r", encoding="latin-1") as file:
            sql_script = file.read()

    cursor.executescript(sql_script)

    print("✅ Vectorized data loaded into memory!")
    return df, conn_mem


# -------------------------------
# 🔍 TRAIN SOFTWARE SIMILARITY MODEL
# -------------------------------

def train_similarity_model(df, knn_save_path="knn_similarity_model.pkl"):
    """Trains kNN model for software similarity."""
    print("🔍 Training kNN model for software similarity...")

    embeddings = np.vstack(df["embedding"].values)
    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(embeddings)

    pickle.dump(knn, open(knn_save_path, "wb"))
    print(f"✅ kNN model saved as {knn_save_path}")
    return knn


# -------------------------------
# 📊 TRAIN EFFORT ESTIMATION MODEL
# -------------------------------

def train_effort_model(df, model_save_path="effort_estimation_model.pkl"):
    """Trains XGBoost for effort estimation using only description embeddings."""
    print("📊 Training XGBoost for effort estimation...")

    X = np.vstack(df["embedding"].values)  # Use only SBERT embeddings
    Y = df["effort"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    print(f"✅ Training complete! MAE: {mae:.4f}, R² Score: {r2:.4f}")

    pickle.dump(model, open(model_save_path, "wb"))
    print(f"✅ Effort estimation model saved as {model_save_path}")
    return model


# -------------------------------
# 🌐 FASTAPI ENDPOINTS
# -------------------------------

app = FastAPI()

@app.post("/predict_effort")
def predict_effort(description: str):
    """Predicts effort estimation using only the software description."""
    print("🖥️ Processing request...")

    knn = pickle.load(open("knn_similarity_model.pkl", "rb"))
    effort_model = pickle.load(open("effort_estimation_model.pkl", "rb"))
    df = pd.read_pickle("repo_embeddings.pkl")

    new_embedding = sbert_model.encode([description])

    distances, indices = knn.kneighbors(new_embedding)
    similar_repos = df.iloc[indices[0]][["owner", "repo", "description", "num_files_dirs", "size_kb"]].to_dict(orient="records")

    predicted_effort = effort_model.predict(new_embedding)[0]

    print("✅ Prediction complete!")
    return {
        "estimated_effort": predicted_effort,
        "similar_projects": similar_repos
    }


# -------------------------------
# 🚀 MAIN EXECUTION FUNCTION
# -------------------------------

def main():
    """Runs the full pipeline with in-memory databases."""
    print("🚀 Starting OSS Effort Estimation Pipeline...")

    # Step 1: Load data into memory
    df, conn = load_data()

    # Step 2: Vectorize descriptions and reload into memory
    df, conn = vectorize_descriptions(df)

    # Step 3: Train similarity model
    # train_similarity_model(df)

    # Step 4: Train effort estimation model
    # train_effort_model(df)

    # Step 5: Start API server
    print("🌐 Starting API...")
    # uvicorn.run(app, host="0.0.0.0", port=8000)


# -------------------------------
# 🎯 RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    main()
