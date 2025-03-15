# Measuring and Verifying Success Rate of Each Algorithm

## **Evaluation Metrics Used**
### 1. **Magnitude of Relative Error (MRE)**
   - Measures the relative size of the difference between the actual and estimated effort value.
   - **Formula**:  
     ```
     MRE = |e' - e| / e
     ```
   - Lower values indicate better estimation accuracy.

### 2. **Mean Magnitude of Relative Error (MMRE)**
   - Averages MRE over multiple projects.
   - **Formula**:  
     ```
     MMRE = (100 / n) * ∑ MRE_i
     ```
   - Lower values indicate better performance.

### 3. **Median Magnitude of Relative Error (MdMRE)**
   - The median of MRE values, reducing sensitivity to outliers.

### 4. **Mean Absolute Residual (MAR)**
   - Measures the mean absolute difference between predicted and true effort values.
   - **Formula**:  
     ```
     MAR = (∑ |e'_i - e_i|) / n
     ```
   - Lower values indicate better predictions.

### 5. **Logarithmic Standard Deviation (LSD)**
   - Measures the standard deviation of log-transformed prediction values.
   - **Formula**:  
     ```
     LSD = sqrt(∑ (ln(e'_i) - ln(e_i))²)
     ```

### 6. **RE*** (Baseline Error Measurement)
   - Measures how much the model reduces variance in residuals compared to actual variance in effort estimates.
   - **Formula**:  
     ```
     RE* = var(residuals) / var(measured)
     ```
   - Lower values indicate better performance.

### 7. **Standardized Accuracy (SA)**
   - Provides a relative performance assessment compared to random guessing.
   - **Formula**:  
     ```
     SA_P = (1 - (MAR_P / MAR_RG)) * 100
     ```
   - Higher values indicate better performance.

### 8. **Effect Size (Cohen’s d, Hedges’ g, Glass’s Δ, and Cliff’s δ)**
   - Statistical tests used to measure the magnitude of differences between predicted and actual effort values.
   - Validates whether differences are significant or occur by chance.

---

## **Validation Methods**
### 1. **Randomized Trials**
   - Divides dataset into training and testing sets randomly.
   - Runs multiple trials (typically 20 iterations) for consistency.
   - Performance metrics are averaged over all trials.

### 2. **k-Fold Cross Validation**
   - Splits data into **k** subsets (commonly k=10).
   - Each subset is used once for testing while others are used for training.
   - Helps evaluate the **generalization capability** of the model.

### 3. **Statistical Significance Tests**
   - **t-Test & Cliff’s δ**: Verifies that estimated effort values differ significantly from random guessing.
   - **Pearson’s Correlation Coefficient**: Measures how strongly **SDEE metrics** correlate with actual development effort.

### 4. **External Validation (Experiment #6)**
   - Involves **professional programmers** evaluating the tool’s predictions.
   - Compares **predicted effort** with **actual effort** from real-world datasets like **COCOMO81**.

---

## **Findings from the Article**
- The proposed method (**DevSDEE**) achieved the **highest standardized accuracy (SA) of 87.26%** in k-fold cross-validation.
- **DevSDEE significantly outperformed traditional models** such as **ATLM, ABE, and LOC straw man estimators**.
- **Developer activity-based metrics** were found to be **strong predictors of effort**.
- **Using project descriptions for similarity detection** led to **better matching of past software efforts** than traditional metadata-based methods.

---


