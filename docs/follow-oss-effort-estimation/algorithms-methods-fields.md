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

