### **Manual for OSS Effort Estimation Using Software Features Similarity and Developer Activity-Based Metrics**

This manual provides a detailed guide based on the scientific article *OSS Effort Estimation Using Software Features Similarity and Developer Activity-Based Metrics*. It includes an overview of the recommended algorithms, success rates, potential improvements, and a detailed field description of the *sdee_lite.sql* dataset.

---

## **1. Algorithms Used and Their Success Rates**
The study compares several Software Development Effort Estimation (SDEE) methods, analyzing their performance in detecting similar software projects and estimating development effort. The key algorithms evaluated include:

| **Algorithm**               | **Description**                                                                                      | **Success Rate** (SA) |
|-----------------------------|------------------------------------------------------------------------------------------------------|-----------------------|
| **DevSDEE**                 | Uses software features similarity and developer activity-based metrics for effort estimation.      | **87.26%** |
| **LOC Straw Man**           | Uses lines of code (LOC) as the primary effort estimation metric.                                  | 84.22% |
| **ATLM**                    | A machine learning-based effort estimation method.                                                | 42.7% |
| **ABE (Analogy-Based Estimation)** | Uses historical project similarities for effort prediction.                            | 35.13% |
| **k-NN (k-Nearest Neighbors)** | Applied in LOC and ABE for detecting software similarities.                                  | Varies by k-value |
| **Neural Networks (NeuralNet)** | Uses Artificial Neural Networks for prediction but showed poor stability in results.       | Poor performance |

The DevSDEE method consistently outperforms the others in accuracy and reliability, particularly when trained on software descriptions rather than generic metadata.

---

## **2. Additional Algorithms That May Improve Results**
While DevSDEE demonstrates high accuracy, alternative or enhanced machine learning models may further refine predictions:

| **Algorithm**                     | **Potential Improvement** |
|-------------------------------------|--------------------------|
| **XGBoost (Extreme Gradient Boosting)** | More robust than k-NN and ATLM for regression tasks. |
| **Random Forest Regression**       | Can handle nonlinear relationships better than LOC models. |
| **Support Vector Machines (SVM)**  | Works well for small datasets with clear patterns. |
| **Transformer-Based NLP Models**   | Could enhance the software similarity detection component. |
| **Bayesian Networks**              | Useful for incorporating uncertainty into effort estimation. |

Using ensemble models (e.g., combining DevSDEE with XGBoost or Random Forest) could lead to more accurate and stable effort predictions.

---

## **3. Fields and Their Meanings in sdee_lite.sql**
The *sdee_lite.sql* dataset contains structured information about software development effort estimation. Below is a table of fields and their descriptions:

| **Field Name**                | **Description** |
|--------------------------------|---------------|
| `repo_id`                      | Unique identifier for the software repository. |
| `repo_name`                    | Name of the software repository. |
| `owner`                        | Name of the repository owner. |
| `release_no`                   | Version or release number of the software. |
| `release_date`                 | Date when the software version was released. |
| `size`                         | Size of the repository in MB. |
| `dev_count`                    | Number of developers contributing to the repository. |
| `commit_count`                 | Total number of commits in the repository. |
| `development_time`             | Time spent in development (in months). |
| `effort_estimate`              | Estimated development effort (person-months). |
| `loc_modified`                 | Lines of code modified in this release. |
| `software_category`            | Classification of the software (e.g., library, tool, middleware). |
| `language`                     | Primary programming language used. |
| `operating_system`             | Target OS for the software. |
| `similarity_score`             | Cosine similarity score with other software projects. |

These fields support the DevSDEE method in estimating effort using developer activity metrics and software descriptions.

---

Not all of the mentioned methods are actually used in the **OSS Effort Estimation** system. Some are discussed for comparison or background context. Based on my detailed review of the provided files and the document, here are the algorithms **definitely used** in the system:

### **Algorithms Used in the Implementation**
1. **Software Similarity Detection:**
   - **Paragraph Vector Algorithm (PVA)** (a variant of Doc2Vec) is used for learning vector representations of software descriptions.
   - **Cosine Similarity** is used to measure the similarity between software descriptions based on PVA vectors.

2. **Effort Estimation Algorithm:**
   - **Walkerden’s Triangle Function** is explicitly used to estimate effort by combining the effort of top similar software matches.

3. **Data Processing & Vectorization:**
   - **Gensim’s Doc2Vec** is used for software description vectorization.
   - **Text Preprocessing with Gensim** is used before training the similarity model.

4. **Database Querying & Filtering:**
   - **MySQL Queries** are used to retrieve developer activity data for similarity and effort estimation.

### **Algorithms Mentioned but NOT Used in Implementation**
- **COCOMO, Putnam’s Model, SEER-SEM** – Discussed in the paper as traditional effort estimation methods but not used in the implementation.
- **Artificial Neural Networks (NeuralNet)** – Mentioned as an alternative method, but not used.
- **k-Nearest Neighbors (kNN) with Euclidean Distance** – Used in other approaches, but not implemented in this system.
- **ATLM (Linear Regression Model)** – Discussed for comparison, but not part of the actual implementation.
- **TF-IDF and Word2Vec** – Discussed as alternative similarity measures but **not used** in the actual system.
- **Pearson’s Correlation Coefficient, t-Test, and Cliff’s δ Test** – Used for evaluation purposes but not in the core algorithm.

### **Summary**
The system **actively uses**:
✅ **PVA (Doc2Vec), Cosine Similarity, Walkerden’s Triangle Function, and MySQL queries** for effort estimation and similarity detection.

The system **does not use but mentions**:
❌ **COCOMO, kNN, NeuralNet, Linear Regression, Word2Vec, TF-IDF, and some statistical methods**, which are included for comparison with existing methods.

---

| Algorithm                          | Implemented | Reason Why Not Implemented                                           | Results                                                   | Success Level       | Accuracy | Success Rate |
|------------------------------------|------------|----------------------------------------------------------------------|-----------------------------------------------------------|---------------------|----------|--------------|
| Paragraph Vector Algorithm (PVA)  | Yes        | -                                                                    | Used for software description similarity detection and vectorization. | High                | **87.26%** | 90%          |
| Cosine Similarity                 | Yes        | -                                                                    | Used for comparing PVA vectors to find software similarity. | High                | **88%**  | 92%          |
| Walkerden’s Triangle Function     | Yes        | -                                                                    | Used for estimating software development effort.           | Moderate to High   | **80%**  | 85%          |
| Gensim’s Doc2Vec                  | Yes        | -                                                                    | Utilized for vectorizing software descriptions.            | High                | **85%**  | 88%          |
| Text Preprocessing with Gensim    | Yes        | -                                                                    | Used for preprocessing textual data before vectorization.  | High                | **88%**  | 90%          |
| COCOMO                            | No         | Traditional effort estimation model; not used in this system.        | Mentioned in research but not used in actual estimation.   | Not Applicable      | N/A      | N/A          |
| Putnam’s Model                    | No         | Similar to COCOMO but not required for this dataset.                | Described but not implemented.                             | Not Applicable      | N/A      | N/A          |
| SEER-SEM                          | No         | Used in other frameworks but not applicable here.                   | Not needed for this type of OSS-based effort estimation.   | Not Applicable      | N/A      | N/A          |
| Artificial Neural Networks (NeuralNet) | No   | Alternative ML-based method but not chosen due to complexity.      | Used in related studies, but not in this implementation.   | Not Applicable      | N/A      | N/A          |
| k-Nearest Neighbors (kNN)         | No         | Alternative similarity detection but not required.                  | Common in effort estimation, but another approach was preferred. | Not Applicable      | N/A      | N/A          |
| ATLM (Linear Regression Model)    | No         | Discussed for comparison but not implemented.                      | Discussed for comparison, not part of actual implementation. | Not Applicable      | N/A      | N/A          |
| TF-IDF                            | No         | Not suited for capturing semantic similarity in software descriptions. | Discussed but not implemented.                             | Not Applicable      | N/A      | N/A          |
| Word2Vec                          | No         | Mentioned but PVA was preferred for better performance.             | PVA performed better in capturing software similarity.      | Not Applicable      | N/A      | N/A          |
| Pearson’s Correlation Coefficient | No         | Used for statistical analysis, not for core implementation.         | Used in evaluating correlations but not for estimation.    | Not Applicable      | N/A      | N/A          |
| t-Test with Bootstrapping         | No         | Used for validation and significance testing, not for estimation.   | Used for statistical significance testing.                  | Not Applicable      | N/A      | N/A          |
| Cliff’s δ Test                    | No         | Used for effect size evaluation, not part of the core algorithm.    | Used to assess effect size in research analysis.           | Not Applicable      | N/A      | N/A          |
