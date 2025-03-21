# 🚀 OSS Effort Estimation Pipeline

This guide provides instructions on how to set up, run, and test the **OSS Effort Estimation API**.

## 📌 Prerequisites

Ensure you have the following installed on your system:
- **Python 3.8+**
- **SQLite** (for database handling)
- **Pip (Python Package Manager)**

---

## 🛠️ Step 1: Install Dependencies

Run the following command to install the required dependencies:

```sh
pip install pandas numpy fastapi sentence-transformers xgboost scikit-learn uvicorn
```

---

## 📂 Step 2: Load & Prepare Data

Ensure that the SQLite database file **`sdee_lite_description.sql`** is in the same directory as the script.

This file contains software repositories' descriptions and metadata. The script will:
1. Load the dataset from **`sdee_lite_description.sql`**.
2. Clean unnecessary data and save it as **`sdee_lite_desc_1.sql`**.
3. Convert descriptions into vector embeddings using **SBERT** and save as **`sdee_lite_desc_vect.sql`**.

---

## 🚀 Step 3: Run the Script

Run the pipeline using the following command:

```sh
python oss_effort_pipeline.py
```

The script will:
- **Load and clean data** from the SQLite database.
- **Convert software descriptions** into vector embeddings using **SBERT**.
- **Train a kNN model** for software similarity detection.
- **Train an XGBoost model** for effort estimation.
- **Start the FastAPI server** for predictions.

---

## 🌐 Step 4: Access the API

Once the script is running, the API will be available at:

```
http://localhost:8000
```

### 🔍 **Test API with cURL**
You can test the API using `curl`:

```sh
curl -X 'POST' \
  'http://localhost:8000/predict_effort' \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "A lightweight AI compression tool"
  }'
```

### 📌 **Expected JSON Response**
```json
{
    "estimated_effort": 45.67,
    "similar_projects": [
        {"owner": "google", "repo": "draco", "description": "A 3D compression library", "num_files_dirs": 350, "size_kb": 2048},
        {"owner": "k0dai", "repo": "density", "description": "Lightweight high-speed compression", "num_files_dirs": 300, "size_kb": 1800}
    ]
}
```

---

## 🧪 Step 5: Running Automated Tests

The script includes **automated test cases** that:
- Run **3 sample descriptions** through the model.
- Print **predictions and explanations** to the console.

To execute the tests manually, modify **`oss_effort_pipeline.py`** to call:

```python
run_tests()
```

---

## 🔄 Step 6: Retraining the Model

If the dataset changes, retrain the model by **deleting the saved models** and running the script again:

```sh
rm knn_similarity_model.pkl effort_estimation_model.pkl
python oss_effort_pipeline.py
```

