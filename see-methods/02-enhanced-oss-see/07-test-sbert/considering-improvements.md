---

## ❓ How Can We Estimate Effort *Before* the Software Exists?

### ✅ **The Goal**:
Estimate the **development effort** of a new software project **based only on its description**, i.e. **before** it's implemented.

But the original paper also uses:
- Developer count
- Development duration
- Code modifications (SLOC)
  
⚠️ Those are all **post-implementation metrics**.

---

## 📄 What the Paper Does

### 🔹 1. **Description-Based Similarity**
- They use software descriptions (e.g., from README files) to find **similar past projects**.
- This part **does not require implementation** — ✅ valid pre-implementation step.

### 🔹 2. **Estimate Using Similar Projects**
Once they find the most similar projects based on description, they:
- Use the **actual effort of those past projects** (which were already implemented)
- Apply **triangle-weighted average** to predict the new project’s effort

> So they’re saying:  
> “Your project looks similar to project A, B, and C → their actual efforts were X, Y, Z → so yours will be about the same.”

✅ **This is reasonable** because you’re not using post-implementation data of the new project — only of similar historical ones.

---

## 🚫 What Would Not Make Sense

If we trained a model like:

```python
Input: [description, dev_count, sloc, etc]
Output: effort
```

…then tried to use that **on a new project** where those features aren’t available yet — it would be cheating or useless.

---

## ✅ What You Can Do Instead

### Option A: **Pure Similarity-Based Estimation (what you're doing now)**
- Use only description → find similar projects
- Use their known effort as proxy
- No cheating — but noisy due to description-effort mismatch

### Option B: **Train ML model using only description vector**
- Input: SBERT vector of description
- Output: effort
- No extra features needed — this is like regression based on semantics

### Option C: **Hybrid: SBERT vector + “early” metadata**
- For new projects where partial metadata exists (e.g. tech stack, team size plans), use them
- Combine that with SBERT to train an ML model
- Example input:
  ```python
  [SBERT vector, planned_dev_count, tech_stack, category]
  ```

---

## 🧠 Key Insight

> The paper’s method works because it leverages **completed past projects** to infer effort for **future projects** that look like them, based on **description alone**.

The "post-implementation" fields are used only for those **historical examples** — not for the new target project.

---

## 🧪 So What Can You Do?

- ✅ Stick with similarity-based estimation using description only
- ✅ Improve accuracy by filtering/weighting better similar projects
- ✅ Or: Train a regressor using SBERT description embeddings → effort

----------------------------------


Thanks for sharing the test results — and you're absolutely right to question them. Here's a breakdown of what the results show and how to improve:

---

## 📉 What's Going Wrong?

From the stats:

- **Mean accuracy**: **-3494%** → this means predictions are wildly off on average.
- **Median accuracy**: **-167%** → half of the predictions are more than double the actual effort in the wrong direction.
- There are even samples with **negative accuracy** — which happens when the estimated effort is far from actual.

This confirms: **the current method (as-is) performs poorly**.

---

## 🔍 Why Is the Accuracy So Poor?

### 1. **Effort Distribution Is Highly Skewed**
- Projects in the dataset may range from tiny scripts (effort ~1) to massive enterprise apps (effort ~1000+).
- Cosine similarity on text alone can't account for this huge variation.

### 2. **SBERT Only Encodes Semantics, Not Scope**
- SBERT understands meaning, **not size or complexity**.
- Two projects that say "REST API backend for shopping cart" and "REST API backend for CRM system" may sound similar but differ massively in scope.

### 3. **Similarity ≠ Effort Correlation**
- Similar wording does not mean similar development workload.

### 4. **Developer Activity Is Missing**
- The original paper uses developer activity (time × #devs).
- This implementation **only uses description → similarity**, and not other metrics like size, commit history, language complexity, etc.

---

## 🛠 Recommendations to Improve Accuracy

### ✅ 1. **Combine SBERT with Additional Features**
Enhance your model input with:
- `sloc_modifications` (lines of code)
- `num_files_dirs`
- `languages` (e.g., Java is more verbose than Python)
- `dev_count`, `active_days`

This enables you to **train a real regression model (e.g. XGBoost or MLP)** using:
```python
features = [sbert_vector + dev_metrics]
label = effort
```

---

### ✅ 2. **Train ML Model Instead of Similarity Heuristic**
Instead of averaging similar efforts:
- Train a model like **XGBoost, RandomForest, or MLPRegressor**
- Input = SBERT vectors + features
- Output = predicted effort

This learns actual **mapping from language → effort**

---

### ✅ 3. **Normalize or Bucket Effort Values**
Try binning effort into **categories** like:
- Low (0–50 person-months)
- Medium (50–200)
- High (200+)

Then run this as a **classification task** instead of regression — easier to get accurate.

---

### ✅ 4. **Clean Extreme Outliers**
Some repos may have erroneous or inconsistent effort values. Trim or Winsorize the top and bottom 1% to reduce noise.

---

### ✅ 5. **Use Log Scale for Effort**
Apply `log1p(effort)` transformation to reduce skew before training or comparing:
```python
df["log_effort"] = np.log1p(df["actual_effort"])
```

---

## 📈 Better Evaluation Metrics

Right now you're using a simple `1 - abs(pred - actual) / actual` as "accuracy" — but it's sensitive to outliers.

Use:
- **MAE** (mean absolute error)
- **MAPE** (mean absolute % error)
- **RMSE**
- **R²** (coefficient of determination)

---

## ✅ Summary

| Action | Benefit |
|--------|---------|
| Add developer metrics | Adds complexity awareness |
| Train ML regressor | Learns non-linear mappings |
| Log-scale effort | Tames extreme values |
| Use better accuracy metrics | More stable evaluation |
| Normalize inputs | Improves learning and similarity quality |

---

perhaps: implement the improved version (SBERT + XGBoost or MLP + structured features)?