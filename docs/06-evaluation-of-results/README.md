## Using PERT and the Beta Distribution for Software Effort Estimation

### 🎯 Overview

To handle the inherent uncertainty in software effort estimation, we use the **Program Evaluation and Review Technique (PERT)** combined with the **Beta distribution**. This approach helps define a probabilistic boundary (or envelope) around effort predictions, rather than relying on a single deterministic value.

---

### 🔍 What is PERT?

PERT estimates the expected effort for a task using three time/effort estimates:

- **Optimistic (O):** The best-case scenario (minimal effort).
- **Most Likely (M):** The most realistic estimate (modal value).
- **Pessimistic (P):** The worst-case scenario (maximum effort).

These values are combined using the Beta distribution formula:

\[
E = \frac{O + 4M + P}{6}
\]

Where:
- `E` is the **expected effort**.

The **standard deviation** and **variance** are calculated as:

\[
SD = \frac{P - O}{6}, \quad \text{Variance} = \left(\frac{P - O}{6}\right)^2
\]

This allows the creation of a confidence interval:

\[
\text{Effort Range (95% CI)} = [E - 2SD, E + 2SD]
\]

---

### 🧠 Why Use This?

- **Quantifies uncertainty** around effort estimates.
- **Captures expert knowledge** with simple inputs (O, M, P).
- Can be combined with **machine learning** or static analysis features for hybrid models.
- Useful for **reporting** and **stakeholder communication**, e.g., showing effort ranges instead of exact numbers.

---

### 📘 Example

Assume we are estimating effort for a module that:

- Optimistic estimate (O) = 3 person-days
- Most likely estimate (M) = 5 person-days
- Pessimistic estimate (P) = 10 person-days

Then:

#### 1. **Expected Effort**:
\[
E = \frac{3 + 4 \cdot 5 + 10}{6} = \frac{33}{6} = 5.5 \text{ person-days}
\]

#### 2. **Standard Deviation**:
\[
SD = \frac{10 - 3}{6} = \frac{7}{6} \approx 1.17
\]

#### 3. **Effort Range (95% confidence)**:
\[
[5.5 - 2 \cdot 1.17,\ 5.5 + 2 \cdot 1.17] \Rightarrow [3.16,\ 7.84] \text{ person-days}
\]

So the expected effort is **5.5 days**, but the actual effort may reasonably fall between **3.16 and 7.84 days**.

---

### 🛠 Integration in Our System

Each subtask in our LLM-based pipeline is tagged with its `O`, `M`, and `P` values. The system then:

- Calculates the **expected effort (E)** and **range** using PERT.
- Displays the result in the UI and final PDF report.
- Optionally feeds `E` and `SD` as features into a machine learning model for enhanced prediction.

---

### 🚀 Future Work

- Automate `O`, `M`, `P` extraction from historical data or expert input via a prompt.
- Combine PERT-based estimates with SBERT-based code similarity scores and static analysis for a hybrid SEE model.
- Extend to full Monte Carlo simulation using Beta distributions for cumulative project effort forecasting.

