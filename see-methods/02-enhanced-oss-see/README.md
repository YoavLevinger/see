Here’s the **refactored version** of your script with **modular functions** for better readability and maintainability. 🚀

---

### **📜 Refactored Python Script**
Save this as **`oss_effort_estimation.py`** and run it.

```python
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
# 🛠️ DATABASE OPERATIONS
# -------------------------------
def clean_old_data(db_path="oss_effort.db"):
    """Removes outdated tables from the database."""
    print("🛠️ Cleaning up old tables...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript("""
        DROP TABLE IF EXISTS soft_desc_pva_vec;
        DROP TABLE IF EXISTS avg_repo_effort;
    """)
    conn.commit()
    conn.close()


def extract_features(db_path="oss_effort.db"):
    """Extracts relevant data from the database."""
    print("📌 Extracting features from database...")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        r.owner, r.repo, r.description, r.num_files_dirs, r.languages, r.size_kb,
        COUNT(DISTINCT c.dev_id) AS dev_count,
        SUM(c.sloc_modifications) AS total_sloc_mod,
        SUM(c.dev_time) AS total_dev_time,
        re.days AS total_days,
        re.effort
    FROM repo_additional_info r
    LEFT JOIN commit_stats c ON r.repo = c.repo AND r.owner = c.owner
    LEFT JOIN release_effort_estimate re ON r.repo = re.repo AND r.owner = re.owner
    GROUP BY r.owner, r.repo;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert categorical `languages` into one-hot encoding
    df = pd.get_dummies(df, columns=["languages"])
    
    return df


# -------------------------------
# 🔥 SOFTWARE SIMILARITY MODEL
# -------------------------------
def train_sbert_embeddings(df, save_path="repo_embeddings.pkl"):
    """Generates SBERT embeddings and saves them."""
    print("🔥 Training SBERT for similarity detection...")
    
    df["embedding"] = df["description"].apply(lambda x: sbert_model.encode(x))
    embeddings = np.vstack(df["embedding"].values)

    # Train kNN model
    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(embeddings)

    # Save embeddings & model
    df.to_pickle(save_path)
    pickle.dump(knn, open("knn_model.pkl", "wb"))

    return knn


# -------------------------------
# ⚡ EFFORT ESTIMATION MODEL
# -------------------------------
def train_effort_model(df, save_path="effort_model.pkl"):
    """Trains the XGBoost effort estimation model."""
    print("⚡ Training XGBoost for effort estimation...")

    # Prepare 02-dataset
    X = df.drop(columns=["owner", "repo", "description", "embedding", "effort"])
    Y = df["effort"]

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, Y_train)

    # Evaluate performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)
    
    print(f"✅ MAE: {mae}, R² Score: {r2}")

    # Save model
    pickle.dump(model, open(save_path, "wb"))

    return model


# -------------------------------
# 🌐 FASTAPI ENDPOINTS
# -------------------------------
app = FastAPI()

@app.post("/predict_effort")
def predict_effort(description: str, num_files_dirs: int, size_kb: float, languages: str):
    """Predicts effort estimation based on software description and repo details."""
    
    # Load trained models
    knn = pickle.load(open("knn_model.pkl", "rb"))
    effort_model = pickle.load(open("effort_model.pkl", "rb"))
    df = pd.read_pickle("repo_embeddings.pkl")

    # Get SBERT embedding for input description
    new_embedding = sbert_model.encode([description])

    # Find similar projects
    distances, indices = knn.kneighbors(new_embedding)
    similar_repos = df.iloc[indices[0]][["owner", "repo", "description"]].to_dict(orient="records")

    # Prepare input for effort estimation
    input_data = pd.DataFrame([[num_files_dirs, size_kb]], columns=["num_files_dirs", "size_kb"])
    predicted_effort = effort_model.predict(input_data)[0]

    return {
        "estimated_effort": predicted_effort,
        "similar_projects": similar_repos
    }


# -------------------------------
# 🚀 MAIN EXECUTION FUNCTION
# -------------------------------
def main():
    """Runs all steps: cleaning, data extraction, model training, and API startup."""
    db_path = "oss_effort.db"

    # Step 1: Clean database
    clean_old_data(db_path)

    # Step 2: Extract fresh 02-dataset
    df = extract_features(db_path)

    # Step 3: Train similarity model
    knn_model = train_sbert_embeddings(df)

    # Step 4: Train effort estimation model
    effort_model = train_effort_model(df)

    # Step 5: Start API server
    print("🌐 Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# -------------------------------
# 🎯 RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    main()
```

---

## **✨ What’s Improved?**
✅ **Code is modularized into functions**  
✅ **Easier debugging and maintenance**  
✅ **FastAPI is well-structured and clear**  
✅ **Separation of Concerns (Database, Models, API)**  

---

## **🚀 How to Run**
### **1️⃣ Install Dependencies**
```sh
pip install pandas numpy fastapi sentence-transformers xgboost scikit-learn uvicorn
```

### **2️⃣ Run the Script**
```sh
python oss_effort_estimation.py
```

### **3️⃣ Test the API**
Use `curl` or **Postman**:
```sh
curl -X 'POST' \
  'http://localhost:8000/predict_effort' \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "A lightweight AI compression tool",
    "num_files_dirs": 350,
    "size_kb": 2048,
    "languages": "Python, C++"
  }'
```

✅ **Example API Response**
```json
{
    "estimated_effort": 45.67,
    "similar_projects": [
        {"owner": "google", "repo": "draco", "description": "A 3D compression library"},
        {"owner": "k0dai", "repo": "density", "description": "Lightweight high-speed compression"}
    ]
}
```


