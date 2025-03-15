# **Manual for Estimating the Best Model to Challenge the Current SEE Model**

## **Introduction**
This is a structured approach to selecting the best algorithm to surpass the current **Software Effort Estimation (SEE) model**, as described in the article *"OSS Effort Estimation Using Software Features Similarity and Developer Activity-Based Metrics."* The goal is to evaluate various machine learning algorithms and determine the one with the highest potential to improve accuracy and efficiency.

---

## **1. Criteria for Selecting a Superior SEE Model**
To surpass the current SEE model, the new algorithm should:
- **Improve accuracy**: Reduce error metrics such as **MRE, MMRE, MAR, and RE\***.
- **Enhance generalization**: Perform well on unseen data via **cross-validation**.
- **Be computationally efficient**: Handle large datasets with minimal processing time.
- **Reduce overfitting**: Implement regularization or ensemble techniques to ensure robustness.

---

## **2. Evaluated Algorithms**
The following algorithms were considered based on their suitability for effort estimation:

| **Algorithm**             | **Estimation (Potential Improvement)** | **Reason** |
|---------------------------|--------------------------------------|------------|
| **XGBoost (Extreme Gradient Boosting)** | ⭐⭐⭐⭐⭐ (Very High) | Optimized version of GBM with **parallelization, regularization, and efficient memory usage**. Likely to outperform existing SEE models. |
| **Random Forests**         | ⭐⭐⭐⭐ (High) | Uses multiple decision trees to reduce overfitting and improve generalization. Good for non-linear relationships. |
| **Support Vector Machines (SVM)** | ⭐⭐⭐⭐ (High) | Effective in high-dimensional spaces, robust against overfitting, and performs well in regression tasks. |
| **Neural Networks**        | ⭐⭐⭐ (Moderate-High) | Can capture complex relationships but **requires extensive tuning and large datasets** to avoid overfitting. |
| **k-Nearest Neighbors (k-NN)** | ⭐⭐ (Moderate) | Effective for small datasets, but computationally expensive and **struggles with complex relationships**. |
| **Linear Regression Variants (Lasso, Ridge, ElasticNet)** | ⭐⭐ (Moderate) | Works well for **linear relationships**, but lacks power for complex software effort estimation. |
| **Bayesian Regression** | ⭐ (Low-Moderate) | Can improve estimation with **probabilistic modeling**, but may struggle with large, complex datasets. |

---

## **3. Best Candidate: XGBoost**
### **Why XGBoost is the Best Choice?**
XGBoost is a **powerful gradient boosting algorithm** that surpasses traditional models like **GBM, ATLM, and ABE** due to:
- **Faster Execution:** Uses **histogram-based split finding** and parallelized execution.
- **Better Generalization:** Incorporates **L1 and L2 regularization** to prevent overfitting.
- **Robust Handling of Missing Data:** Automatically detects missing values.
- **Built-in Cross-Validation & Early Stopping:** Optimizes hyperparameters efficiently.

### **Comparison with Existing SEE Models**
| **Model** | **Accuracy (SA %)** | **Computational Efficiency** | **Overfitting Control** |
|-----------|---------------------|-----------------------------|--------------------------|
| **ATLM**  | 42.7%               | Moderate                    | Low                      |
| **ABE**   | ~50-60%             | Low                         | Moderate                  |
| **LOC Straw Man** | 87.26%       | Low                         | Poor                     |
| **DevSDEE** | 87.26%             | High                        | High                     |
| **XGBoost (Proposed)** | **>90% (Expected)** | **Very High** | **Very High** |

