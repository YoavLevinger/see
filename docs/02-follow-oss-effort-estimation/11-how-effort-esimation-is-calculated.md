# Effort Estimation Calculation

## Overview  
The effort estimation in the referenced article is calculated using a combination of **software similarity detection** and **developer activity-based metrics**. The methodology involves the following key steps:

---

## 1. Developer Activity Information as Metrics  
- The effort estimation leverages **developer activity data** extracted from version control system (VCS) repositories.  
- Key metrics include:  
  - **Source Lines of Code (SLOC) added, deleted, and modified**  
  - **Developer count**  
  - **Time spent on development**  
- The total effort spent on a project (`e_r`) is defined as:

e_r = |D_r| × t_r


where:  
- `|D_r|` is the number of developers  
- `t_r` is the total time spent on development  

---

## 2. Software Similarity Detection Model (PVA-Based)  
- A **machine learning model trained using Pairwise Vector Averaging (PVA)** is used to determine functionally similar software projects based on their descriptions.  
- This model predicts software projects that closely match a given new software requirement.  

---

## 3. Effort Estimation Formula  
- Once similar projects are identified, their effort values are aggregated using **Walkerden’s triangle function**, which weights the effort of the most similar projects as:

Effort(z) = (3a + 2b + 1c) / 6


where:  
- `a, b, c` are the effort values of the top three most similar software projects.  

---

## 4. Comparison with Other Methods  
The proposed model is compared against existing effort estimation methods, including:  
- **ATLM (Automatically Transformed Linear Model)** - a regression-based approach.  
- **Analogy-Based Estimation (ABE)** - uses k-nearest neighbors (kNN) and Euclidean distance.  
- **LOC "straw man" estimator** - uses lines of code as a predictor.  
- The proposed model achieves **higher accuracy (87.26%)** compared to traditional models like ATLM (42.7%).  

---

## 5. Dataset and Implementation  
- The dataset consists of approximately **13,000 GitHub repositories across 150 software categories**.  
- Data is collected using **GitHub’s API**, filtering repositories based on:  
- **Size constraint**: More than 5MB.  
- **Activity constraint**: Updated at least once in the last three years.  
- **Popularity constraint**: More than 500 stars.  
- The dataset is used to develop a **software effort estimation tool**.  

---

## Conclusion  
This approach provides a **data-driven and automated** method for software effort estimation, leveraging **developer activity metrics** and **software description similarity** for higher accuracy. It outperforms expert-based and traditional statistical models in **accuracy and efficiency**.  

---

# Usage of SLOC in Effort Estimation

## 1. Developer Activity Representation
- SLOC is used to track the number of lines of code **added, deleted, or modified** in a project.
- This information is derived from **version control system (VCS) repositories**, such as GitHub.
- The SLOC data is stored in the `commit_stats` table along with the associated commit and developer IDs.

---

## 2. Effort Computation
- The **release-level effort estimates** are computed by aggregating commit-level SLOC modifications over time.
- The total number of SLOC modifications in a repository (`SLOCr_m`) is recorded to capture the intensity of developer activity.

---

## 3. Correlation with Development Effort
- An experiment was conducted to determine the correlation between SLOC modifications and development effort.
- The **Pearson correlation coefficient** results:
- **Development time:** Highest correlation (`0.799`).
- **Developer count:** Moderate correlation (`0.644`).
- **SLOC modifications:** Weak correlation (`0.065`).
- This indicates that while SLOC contributes to effort estimation, it is not as strong a predictor as development time or developer count.

---

## 4. Integration in Effort Estimation Models
- Traditional parametric models, such as **COCOMO**, use SLOC as a core input variable to estimate software development effort.
- The effort estimation formula typically follows:

E = A + B * (SLOC)^C


where `A`, `B`, and `C` are empirically derived constants.
- However, the study found that **developer count and time spent on development** were more influential than SLOC alone.
