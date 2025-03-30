# GitHub Repository Effort Estimation via Static Code Analysis

This project implements a scientifically grounded tool to estimate software development effort (in hours and person-months) from static code repositories using Python. The core of the estimation is based on a hybrid model: combining COCOMO II effort estimation with complexity analysis.

---

## 📊 Summary of the Estimation Workflow

### Step-by-step process:

1. **Extract code metrics** from a GitHub repository:
   - Lines of code (LOC)
   - Cyclomatic complexity
   - Number of functions
   - Git history (commits, contributors, etc.)

2. **Calculate base effort** using the **COCOMO II** power-law size model:
   - Effort (person-months) = `2.94 × (KLOC ^ 1.10)`

3. **Adjust effort** using a **complexity factor** derived from average cyclomatic complexity.

4. **Convert to hours** using standard factor (1 person-month = 152 hours).

---

## 🧮 Effort Estimation Formula

```text
Effort = 2.94 × (KLOC ^ 1.10) × ComplexityFactor
```

- **KLOC** = thousands of lines of code
- **ComplexityFactor** = adjustment based on average cyclomatic complexity

This formula is grounded in the COCOMO II model (Boehm et al.) and adapted to support different complexity calibration strategies.

---

## 🔀 Complexity Factor Modes

You can toggle between 3 different scientifically inspired complexity adjustment strategies:

### 1. **Power-based adjustment (default)**
```python
ComplexityFactor = (AvgCC / 10) ** 0.3
```
- Inspired by exponential cost drivers in COCOMO
- AvgCC = Average cyclomatic complexity per function
- Baseline complexity = 10

### 2. **Linear adjustment**
```python
ComplexityFactor = 1 + 0.05 * (AvgCC - 10)
```
- Adds 5% effort for each point of CC above 10
- Subtracts effort for complexity < 10

### 3. **COCOMO-style lookup table**
| AvgCC Range | Complexity Rating | Factor |
|-------------|-------------------|--------|
| <5          | Very Low          | 0.75   |
| 5–9         | Low               | 0.88   |
| 10–14       | Nominal           | 1.00   |
| 15–19       | High              | 1.15   |
| ≥20         | Very High         | 1.30   |

- These values reflect the COCOMO II complexity cost driver.

Use the `complexity_mode` parameter to switch between them.

---

## 🛠️ Example Python API Usage

```python
repos = [("psf", "requests"), ("pallets", "flask")]
evaluate_multiple_repos(repos, complexity_mode="lookup")
```

Available modes:
- `"power"` — power function (default)
- `"linear"` — linear adjustment
- `"lookup"` — discrete complexity categories

---

## 📘 References

- Barry Boehm. *Software Engineering Economics* (COCOMO model)
- Lavazza et al. (2023). *Evaluation of Cognitive Complexity as Effort Proxy*
- Mišić and Tešić (1998). *Effort Estimation using OO Metrics*
- Rashid et al. (2025). *Hybrid COCOMO + ANN Models*

---

## ✅ Benefits of This Approach

- Fully automatic: uses static code analysis only
- Scientifically backed: all formulas sourced from academic research
- Works across many languages (via Lizard)
- Easily extendable to other metrics (e.g. Halstead, method count)

---

## 📂 Outputs

- Printed metric summary (GitHub-style markdown table)
- CSV file with all metric values and estimations
- Bar chart dashboard of estimated efforts
- Optional log file with full run output

---


