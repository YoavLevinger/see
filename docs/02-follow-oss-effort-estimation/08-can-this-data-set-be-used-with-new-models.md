### **How to Retrain a Model When Only Vectorized Data Exists in the Database?**

If the database only **stores vectorized representations** (i.e., **numerical vectors**) and **not the original text descriptions**, retraining a new `Doc2Vec` model becomes challenging because **Doc2Vec requires raw text** for training.

However, we can still explore **three possible approaches** to retrain or improve the model:

---

## **1. Checking If the Original Text is Available**
### **🔹 First Step: Verify if Text Exists in Another Table**
Before assuming that the raw descriptions are lost, check if there is another table that **stores the original project descriptions**.

**Check these possibilities:**
```sql
SHOW TABLES;  -- Lists all tables in the database
DESC repo_info_pv_vec;  -- Describes the schema of the table
SELECT * FROM repo_info_pv_vec LIMIT 5;  -- Checks if a description column exists
```
- If a column like `project_description` or `text_data` exists, then **we can use that** to retrain the model.

---

## **2. Reverse Engineering the Training Data**
### **🔹 Option A: Recover Text from a Backup or Logs**
If you previously stored text descriptions before vectorizing them, they might be:
- **Stored in log files.**
- **Cached in previous database backups.**
- **Saved in another system (e.g., the frontend input forms, API logs, etc.).**

---

### **🔹 Option B: Use a Pre-Trained Embedding Model to Approximate the Text**
If no raw text is available, we can **use reverse embeddings** to approximate possible text representations.

**How?**
1. **Find the nearest neighbors** for each vector.
2. **Use a pre-trained text embedding model** (like OpenAI’s GPT, BERT, or FastText) to generate words that match the vectors.
3. **Manually validate** the generated text.

Example:
```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load stored vectors from the database
stored_vectors = np.load("vectorized_data.npy")  # Assuming we can extract these vectors

# Load a pre-trained model (e.g., OpenAI’s embedding API or Sentence-BERT)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Find the closest known descriptions
nbrs = NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(stored_vectors)
query_vector = model.encode(["AI chatbot for customer support"])  # Example text
distances, indices = nbrs.kneighbors([query_vector])

print("Closest stored descriptions (approximated):", indices)
```
- This **does not perfectly recover the original descriptions** but provides **similar text descriptions** that we can use.

---

## **3. Retraining a New Model from Scratch**
If **no text data can be recovered**, the only option is to **retrain a new model** using fresh data.

### **🔹 Option A: Collect New Data**
If we can't extract past data, we can:
- **Manually gather open-source project descriptions** (e.g., GitHub, Kaggle, Stack Overflow).
- **Ask users to re-enter project descriptions**.

### **🔹 Option B: Use Open Data Sources**
We can train a new **Doc2Vec model** on public datasets:
- **GitHub Repositories Dataset** (available via API)
- **Papers with Code** (ML research + code implementations)
- **Kaggle Software Projects Dataset**

Example:
```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Example 02-dataset
new_text_data = [
    "An AI chatbot for customer support using NLP",
    "A cloud-based image recognition system",
    "A distributed database for high-performance computing"
]

tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(new_text_data)]

# Train a new Doc2Vec model
new_model = Doc2Vec(vector_size=50, epochs=10, min_count=1, workers=4)
new_model.build_vocab(tagged_data)
new_model.train(tagged_data, total_examples=new_model.corpus_count, epochs=new_model.epochs)

# Save the new model
new_model.save("new_doc2vec.model")
```
✅ **This creates a new model without needing old text descriptions.**

---

## **4. Alternative Approach: Store New Text Descriptions Going Forward**
If the system **only stores vectors**, modify it so that **new project descriptions** are stored alongside their vectors in the future.

**Modify the database schema:**
```sql
ALTER TABLE repo_info_pv_vec ADD COLUMN project_description TEXT;
```
Then, **store both the raw text and the vector** when processing new inputs.

```python
def store_project(description):
    vector = model.infer_vector(word_tokenize(description.lower()))
    db.execute("INSERT INTO repo_info_pv_vec (project_description, vector) VALUES (%s, %s)", (description, vector))
```
🔹 **This prevents future data loss.**

---

## **Final Summary: What to Do Next**
| **Situation** | **Solution** |
|--------------|--------------|
| **Raw text is available in another table** | Extract and retrain Doc2Vec |
| **Raw text is missing but logs exist** | Recover descriptions from logs or backups |
| **Only vectorized data is available** | Use **reverse embeddings** to approximate text |
| **No text or logs exist** | Train a **new model** from external datasets |
| **Prevent future issues** | Modify the database to store **both text & vectors** |

