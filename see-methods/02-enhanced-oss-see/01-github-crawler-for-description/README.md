Absolutely! Here's how the `effort_score` and `effort_months` are calculated in your script:

---

### 🧠 **Conceptual Basis**

The effort estimation is based on a **developer activity model** from the paper:

> **Effort = Number of Developers × Development Time**

---

### 📊 **Implementation in the Script**

#### 1. **effort_score**
```python
effort = developer_count * time_range_days
```

- **`developer_count`**: Number of unique contributors to the repository  
  → Retrieved via GitHub REST API endpoint:
  ```
  https://api.github.com/repos/{owner}/{repo}/contributors?anon=true
  ```

- **`time_range_days`**: The difference between the **first** and **last** commit dates  
  → Estimated from the first 100 commits:
  ```python
  last_date = datetime.strptime(commits[0]['commit']['committer']['date'], ...)
  first_date = datetime.strptime(commits[-1]['commit']['committer']['date'], ...)
  time_range_days = (last_date - first_date).days
  ```

✅ So, `effort_score` is expressed in **person-days**

---

#### 2. **effort_months**
```python
effort_months = round(effort / 30.0, 2)
```

- Converts **effort_score (person-days)** into **person-months**
- Assumes **1 month = 30 days** (a typical approximation in effort estimation)

✅ So, `effort_months` = `effort_score / 30`

---

### 📦 Summary Table

| Metric          | Description                                      | Unit         | Source             |
|------------------|--------------------------------------------------|---------------|---------------------|
| `developer_count` | Number of contributors                          | Integer       | GitHub REST API     |
| `time_range_days` | Time from first to last commit                  | Days          | GitHub commit dates |
| `effort_score`    | `developers × time_range_days`                  | Person-days   | Derived             |
| `effort_months`   | `effort_score ÷ 30`                             | Person-months | Derived             |

---


Great question! GitHub exposes a rich set of metadata through its **REST** and **GraphQL** APIs. Here's a list of **metadata fields you can extract** that are specifically useful for **effort estimation**, and whether they include things like the **first commit** or **first release date**.

---

## 🧩 **Effort-Related Metadata Available from GitHub**

### ✅ **Repository-level Metadata**
| Metric | Description | Available via | Usefulness |
|--------|-------------|----------------|------------|
| `created_at` | Repo creation timestamp | REST / GraphQL | Baseline time |
| `pushed_at` | Last push (commit) time | REST / GraphQL | Recency |
| `disk_usage` | Repo size (in KB) | GraphQL | Project size proxy |
| `language` | Primary language | REST | Classification |
| `languages` | All languages used + size per language | GraphQL | Language complexity |
| `forks_count` | Number of forks | REST | Popularity metric |
| `stargazers_count` | Stars | REST | Popularity |
| `subscribers_count` | Watchers | REST | Community interest |
| `open_issues_count` | Open issues | REST | Maintenance burden |
| `license` | License type | REST | Licensing factor |

---

### ✅ **Commit & Contributor Metadata**
| Metric | Description | How to Get | Use |
|--------|-------------|------------|-----|
| `first_commit_date` | Date of the first commit | ✅ Yes, via paginated `/commits` REST API | ⏱ Start of dev |
| `last_commit_date` | Most recent commit | ✅ Yes (first item from `/commits`) | ⏱ End of dev |
| `commit_count` | Total number of commits | ✅ Yes, via stats or manual counting | Effort indicator |
| `contributors` | List of all contributors | ✅ Yes (`/contributors?anon=true`) | 👤 Team size |
| `commit frequency` | Weekly commit stats | ✅ Yes (`/stats/commit_activity`) | ⏱ Active effort |
| `code_frequency` | Weekly added/deleted LOC | ✅ Yes (`/stats/code_frequency`) | 📈 SLOC metrics |

---

### ✅ **Release Metadata**
| Metric | Description | How to Get | Use |
|--------|-------------|------------|-----|
| `first_release_date` | Timestamp of the first release | ✅ Yes (`/releases`) | ⏱ Start of public delivery |
| `latest_release_date` | Timestamp of latest release | ✅ Yes | Delivery cadence |
| `release_count` | Number of releases | ✅ Yes | Maturity indicator |
| `pre-releases` | Whether a release is marked pre-release | ✅ Yes | Stability signal |

---

### ✅ **Issue & PR Activity**
| Metric | Description | How to Get | Use |
|--------|-------------|------------|-----|
| `issue_opened_date` | Date of first issue | ✅ Yes | Support demand |
| `issue_closed_date` | Time to close | ✅ Yes | Responsiveness |
| `pull_requests` | Count, merge time, etc. | ✅ Yes | Dev process complexity |

---

## 🔍 Can You Get:
| Metric                          | Possible? | How |
|---------------------------------|-----------|-----|
| **Date of first commit**        | ✅ Yes    | Sort `/commits` ascending or iterate through pages to last item |
| **Date of first release**       | ✅ Yes    | `/releases` sorted by `created_at` or `published_at` |

---

## 🛠 Pro Tip: Combining for Effort Estimation
You can derive rich models using:
- `effort = dev_count × (last_commit - first_commit)`
- Enhance with:
  - LOC added/deleted (from `code_frequency`)
  - Release intervals (time from first commit to first release)
  - Developer churn (how often contributors change)

---


