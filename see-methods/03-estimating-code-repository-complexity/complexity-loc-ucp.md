
# Code-Only Effort Estimation Model (COEEM) – Metric Weights Justification

## 📘 Context from Sharma & Chaudhary (2023)

In their study *Prediction of Software Effort by Using Non-Linear Power Regression for Heterogeneous Projects*, Sharma & Chaudhary (2023) emphasize:

- LOC and Use Case Points (UCP) provide strong predictors when combined.
- Non-linear models outperform traditional linear ones.
- Power-regression based estimators minimize estimation error (MMRE).

These insights support a complexity-weighted, non-linear model for code-only effort estimation.

---

## 🧠 Why Each Metric is Weighted This Way

| Metric | Symbol | Weight | Academic Justification |
|--------|--------|--------|--------------------------|
| Lines of Code | `LOC` | 1.0 | Base size metric; useful but less predictive alone. Sharma & Chaudhary (2023) show it requires structural context. |
| Cyclomatic Complexity | `CC̄` | 2.0 | Strong indicator of branching logic and fault proneness. Supported by Lavazza et al. (2023). |
| Halstead Volume | `HV̄` | 1.0 | Reflects logical size and mental workload. Balanced weight due to syntax dependence. |
| Cognitive Complexity | `CoCō` | 1.5 | Represents code readability and understandability. Moderate empirical support. |
| Function Count | `NF` | 1.0 | Linked to modularity and object-oriented structure. Mišić & Tešić (1998). |
| AST Depth | `ASTd̄` | 1.0 | Tree depth reflects code hierarchy and structure. Supported by Phan et al. (2018). |

---

## 📏 Composite Complexity Formula

```math
C_{comp} = w_1 \cdot \frac{LOC}{1000} + w_2 \cdot \overline{CC} + w_3 \cdot \overline{HV} + w_4 \cdot \overline{CoCo} + w_5 \cdot \log(NF + 1) + w_6 \cdot \overline{ASTd}
```

Default Weights:
- `w₁=1`, `w₂=2`, `w₃=1`, `w₄=1.5`, `w₅=1`, `w₆=1`

---

## ⏳ PERT-Wrapped Effort Formula

```math
Effort_{days} = \frac{
0.75 \cdot C_{comp}^{0.85} + 4 \cdot C_{comp} + 1.25 \cdot C_{comp}^{1.15}
}{6}
```

This envelope models uncertainty using PERT (Program Evaluation Review Technique) aligned with power regression trends in Sharma & Chaudhary (2023).

---

## 📚 References

- Sharma, A., & Chaudhary, N. (2023). *Prediction of Software Effort by Using Non-Linear Power Regression*. Procedia Computer Science, 218, 1601–1611.
- Lavazza, L., et al. (2023). *An empirical evaluation of the 'Cognitive Complexity' measure*. Journal of Systems & Software, 197, 111561.
- Abbad-Andaloussi, A. (2023). *On the relationship between source-code metrics and cognitive load*. The Journal of Systems & Software, 198, 111619.
- Kaur, L., & Mishra, A. (2019). *Cognitive complexity as a quantifier of version to version Java-based source code change*. Information and Software Technology, 106, 31–48.
- Mišić, V. B., & Tešić, D. N. (1998). *Estimation of effort and complexity: An object-oriented case study*. The Journal of Systems and Software, 41(2), 133–143.
- Phan, A. V., et al. (2018). *Automatically classifying source code using tree-based approaches*. Data & Knowledge Engineering, 114, 12–25.

---