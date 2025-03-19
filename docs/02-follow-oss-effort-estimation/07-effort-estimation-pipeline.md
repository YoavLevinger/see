
## **1. Overview of Effort Estimation Pipeline**
When a user provides a **new product description**, the system follows these steps:

1. **User Input (Django Forms)**
   - A new product description and feature set are entered through the UI (`forms.py`).
   - `ProductForm` and `FeatureForm` are used to capture user input.

2. **Processing in `views.py`**
   - The **`product_create_view`** extracts the input data and calls `fetch_sim_executor()` from `fetchSimResults.py`.

3. **Similarity Calculation (`fetchSimResults.py`)**
   - The input text is **vectorized** using a pre-trained `Doc2Vec` model.
   - The **cosine similarity** between this vector and stored repository vectors is calculated.
   - **Top-k similar software projects** are retrieved.

4. **Effort Estimation (`fetch_effort_est()`)**
   - Historical **developer effort metrics** (number of devs, active days, modified LOC) are retrieved for the top similar projects.
   - An **average effort** is computed based on a weighted formula.

---

## **2. Detailed Walkthrough**
### **Step 1: User Input**
- The **user submits a product description, programming language, software type, and features** via a Django form (`forms.py`).
- Example input:
  ```python
  {
    "title": "AI Chatbot",
    "prod_description": "A chatbot that assists with customer service.",
    "prog_lang": "Python",
    "soft_type": "Machine Learning",
    "ext_tool_support": "TensorFlow",
    "os_support": "Windows, Linux",
    "features": ["Natural Language Processing", "Voice Recognition"],
    "feature_description": ["Understands human language", "Supports voice commands"]
  }
  ```

### **Step 2: View Processing (`views.py`)**
- The `product_create_view(request)` processes the form and calls:
  ```python
  categoryList, ownerList, repoList, devCountList, totlMonthsList, modLOCList, effortList, simScores, avgEffort = fetch_sim_executor(prod_title, prod_desc, lang, sub_sft_type, tool, os, features, feature_descs)
  ```
- This calls `fetch_sim_executor()` in `fetchSimResults.py`.

### **Step 3: Similarity Computation (`fetchSimResults.py`)**
#### **3.1 Vectorizing the Input**
```python
model = Doc2Vec.load("products/backend/models/pv_doc2vec_effort_est_info_e_10_v_50.model")
refDocPath = 'products/backend/refVecs/gen_sim_ref_vec_e_10_v_50.txt'
refDoc = readFile(refDocPath)
```
- The **Doc2Vec model** (`pv_doc2vec_effort_est_info_e_10_v_50.model`) is loaded.
- `refDoc` is a **reference document vector** used as a baseline.

#### **3.2 Computing Similarity**
```python
cos_sim = vectorize(feature, model, refDoc)
cosSortKeys = get_abs_sorted_keys(cosSimMap, cos_sim)
favMatches = fetch_top_K_match_results(cosSortKeys, cosSimMap, ownerRepoList)
```
- **Input text** is converted into a vector.
- **Cosine similarity** is computed against stored software repositories.
- **Top 10 closest matches** are selected.

---

### **Step 4: Fetching Effort Metrics**
#### **4.1 Fetching Developer Effort Data**
For each of the **top similar repositories**, the function `fetch_effort_est()` retrieves:
```python
query = "SELECT COUNT(DISTINCT devId), SUM(activDays), SUM(modLOC), SUM(effort) FROM release_wise_effort WHERE owner = %s AND repo = %s;"
```
- **Developer count**
- **Total active development days**
- **Lines of code modified**
- **Total effort spent**

#### **4.2 Converting to Time Estimation**
```python
totlMonthsList.append(round(float(fetch_days_count(owner, repo)) / 30, 4))
effortList.append(round(float(row[3]), 4))
```
- Converts total development days into **months**.
- The **effort metric** is extracted.

#### **4.3 Computing Final Effort Estimation**
```python
newEffortList = []
for zz in range(0, 5):
    newEffortList.append(round(float(devCountList[zz]) * totlMonthsList[zz], 4))
effortList = newEffortList
avgEffort = (5 * effortList[0] + 4 * effortList[1] + 3 * effortList[2] + 2 * effortList[1] + effortList[0]) / 15
```
- A **weighted sum** method is used:
  - The **most similar project contributes more** weight.
  - The final **average effort** is computed.

---

## **3. Final Output**
The system returns:
```python
return categoryList[:5], ownerList[:5], repoList[:5], devCountList[:5], totlMonthsList[:5], modLocList[:5], effortList[:5], simScores[:5], avgEffort
```
- **Category** of the similar projects
- **Repository Owners**
- **Repository Names**
- **Developer Count**
- **Total Development Time (Months)**
- **Modified Lines of Code**
- **Effort Estimates**
- **Similarity Scores**
- **Final Average Effort Estimate**

This is then displayed to the user in the **Django UI**.

---

## **4. Example Output**
If a user inputs:
```python
{
  "title": "AI Chatbot",
  "prod_description": "A chatbot for customer support.",
  "prog_lang": "Python",
  "soft_type": "Machine Learning",
  "features": ["NLP", "Voice Recognition"],
}
```
The system may return:
```python
{
  "similar_projects": [
    {"category": "Machine Learning", "repo": "chatbot1", "owner": "companyA", "dev_count": 10, "months": 8, "effort": 80},
    {"category": "AI Systems", "repo": "chatbot2", "owner": "companyB", "dev_count": 8, "months": 6, "effort": 48},
    {"category": "Conversational AI", "repo": "chatbot3", "owner": "companyC", "dev_count": 12, "months": 9, "effort": 108}
  ],
  "estimated_effort": 78.67
}
```
This means that a new **AI Chatbot** would likely require **around 79 developer-months** based on similar projects.

---

## **5. Summary**
1. **User enters software details via Django forms.**
2. **Doc2Vec vectorizes the input and finds similar projects using cosine similarity.**
3. **Database queries fetch effort estimation data (developer count, modified LOC, active days).**
4. **Effort estimation is calculated using weighted averaging.**
5. **The system returns an effort estimate based on past projects.**
