# OSS Effort Estimation Using Software Features Similarity and Developer Activity-Based Metrics

This repository contains resources and documentation related to **Software Development Effort Estimation (SDEE)**, focusing on leveraging **developer activity metrics** and **software similarity detection**. The approach is based on the research by **Ritu Kapur and Balwinder Sodhi** from the Indian Institute of Technology Ropar.

## 📄 Research Paper
The primary reference for this repository is:
- **Kapur, R., & Sodhi, B. (2022).** *OSS Effort Estimation Using Software Features Similarity and Developer Activity-Based Metrics.* ACM Transactions on Software Engineering and Methodology, 31(2), Article 33.  
  📌 **DOI:** [10.1145/3485819](https://doi.org/10.1145/3485819)  
  📌 **Zenodo Repository:** [https://doi.org/10.5281/zenodo.5095723](https://doi.org/10.5281/zenodo.5095723)

---

## 📂 Folder Structure & File Descriptions

### 📌 **Main Documents**
- **[`01-OSS Effort Estimation - Software Features Similarity and Developer Activity.md`](01-OSS Effort Estimation%20-%20Software%20Features%20Similarity%20and%20Developer%20Activity.md)**  
  📖 Overview of the methodology for effort estimation, focusing on developer activity and software similarity detection using the **Paragraph Vector Algorithm (PVA)**.

- **[`02-evaluation-metrics.md`](02-evaluation-metrics.md)**  
  📊 Defines **evaluation metrics** used to assess effort estimation accuracy, including **Magnitude of Relative Error (MRE), Mean Absolute Residual (MAR), and Standardized Accuracy (SA)**.

- **[`03-algorithms-methods-fields.md`](03-algorithms-methods-fields.md)**  
  📚 Lists the **algorithms** used for effort estimation, their **success rates**, and a **description of dataset fields** in `sdee_lite.sql`.

- **[`04-elaborate-on-algorithms.md`](04-elaborate-on-algorithms.md)**  
  🔍 Detailed explanations of **DevSDEE, LOC Straw Man, ATLM, and ABE**, along with SQL-based implementation strategies.

- **[`05-estimating-the-best-model-to-challenge-the-current-see-model.md`](05-estimating-the-best-model-to-challenge-the-current-see-model.md)**  
  🏆 Selection of the **best model to improve the Software Effort Estimation (SEE) system**, with **XGBoost identified as the top choice**.

- **[`06-selecting-algorithm.md`](06-selecting-algorithm.md)**  
  📌 Discussion on how **SDEE dataset** (`sdee_lite.sql`) can be used to optimize **effort estimation models**, including Bayesian Networks and Machine Learning models.

---

## 🗃️ **Dataset Descriptions**
- **[`data-description.md`](data-description.md)**  
  📋 Structure and explanation of dataset tables, including:
  - `release_info`: Software release metadata.
  - `commit_stats`: Developer activity logs.
  - `release_effort_estimate`: Effort estimation records.
  - `soft_desc_pva_vec`: Vectorized software descriptions.
  - `avg_repo_effort`: Aggregated effort estimates.

- **[`vectorized-description-data.md`](vectorized-description-data.md)**  
  🔢 Explanation of how **software descriptions are converted into vectorized data** using the **Paragraph Vector Algorithm (PVA)**.

---

## 📜 **Credit & Acknowledgment**
This work is based on the research paper:

📖 **"OSS Effort Estimation Using Software Features Similarity and Developer Activity-Based Metrics"**  
✍️ *Ritu Kapur and Balwinder Sodhi*  
🏫 **Indian Institute of Technology Ropar**  

Published in **ACM Transactions on Software Engineering and Methodology, 2022**.  
🔗 DOI: [10.1145/3485819](https://doi.org/10.1145/3485819)  
🔗 Zenodo Repository: [https://doi.org/10.5281/zenodo.5095723](https://doi.org/10.5281/zenodo.5095723)  

If you use this repository or its contents, please cite their work accordingly.

---
