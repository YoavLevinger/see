## 📌 **Tables and Fields in SDEE Dataset**

### 1️⃣ **`release_info`**
   - **Description**: Stores information about software releases.
   - **Fields**:
     - `repo` (TEXT) – Repository name.
     - `owner` (TEXT) – Repository owner.
     - `release_no` (INTEGER) – Release number.
     - `release_date` (DATE) – Date of the release.
     - `size` (INTEGER) – Size of the release.

### 2️⃣ **`commit_stats`**
   - **Description**: Stores developer activity data related to commits.
   - **Fields**:
     - `commit_id` (TEXT) – Unique identifier for a commit.
     - `dev_id` (TEXT) – Developer ID.
     - `repo` (TEXT) – Repository name.
     - `owner` (TEXT) – Repository owner.
     - `dev_time` (FLOAT) – Time spent by the developer on the commit.
     - `sloc_modifications` (INTEGER) – Source lines of code modified.
     - `productivity_index` (FLOAT) – Developer productivity metric.
     - `skill_factor` (FLOAT) – Skill factor of the developer.

### 3️⃣ **`release_effort_estimate`**
   - **Description**: Stores release-level effort estimates.
   - **Fields**:
     - `repo` (TEXT) – Repository name.
     - `owner` (TEXT) – Repository owner.
     - `min_release_id` (INTEGER) – ID of the earliest considered release.
     - `max_release_id` (INTEGER) – ID of the latest considered release.
     - `start_release_date` (DATE) – Start date of release period.
     - `end_release_date` (DATE) – End date of release period.
     - `days` (INTEGER) – Number of days in the considered release period.
     - `dev_count` (INTEGER) – Number of developers contributing.
     - `effort` (FLOAT) – Estimated development effort.

### 4️⃣ **`soft_desc_pva_vec`**
   - **Description**: Stores vectorized software descriptions and similarity scores.
   - **Fields**:
     - `category` (TEXT) – Software category (e.g., library, framework).
     - `owner` (TEXT) – Repository owner.
     - `repo` (TEXT) – Repository name.
     - `pv_vec` (BLOB/TEXT) – Precomputed vector representation of software description.
     - `cos_sim` (FLOAT) – Cosine similarity score with reference vector.

### 5️⃣ **`avg_repo_effort`**
   - **Description**: Contains aggregated effort estimates for repositories.
   - **Fields**:
     - `owner` (TEXT) – Repository owner.
     - `repo` (TEXT) – Repository name.
     - `devCount` (INTEGER) – Number of developers involved.
     - `activDays` (INTEGER) – Days of active development.
     - `totlDays` (INTEGER) – Total days since project start.
     - `modLOC` (INTEGER) – Modified lines of code.
     - `daf` (FLOAT) – Developer Activity Factor.
     - `effort` (FLOAT) – Estimated effort required for development.

## 🔎 **Key Observations**
- **Relation Between Tables**:
  - `release_info` and `commit_stats` track software changes over time.
  - `soft_desc_pva_vec` stores **vectorized** software descriptions, making it useful for similarity-based models.
  - `release_effort_estimate` and `avg_repo_effort` provide the **ground truth** for effort estimation.

- **Potential Issues**:
  - `repo` and `owner` fields must match between tables for proper merging.
  - `pv_vec` is already vectorized, making it suitable for similarity-based ML models.

- **Useful for Modeling**:
  - Cosine similarity (`cos_sim`) allows for **text-based similarity** approaches.
  - Developer activity (`devCount`, `activDays`, `modLOC`) supports **traditional regression models**.


# 📊 Software Effort Estimation Dataset Schema

## 📌 Table: `release_info`
This table contains metadata about software releases.

| **Field**         | **Type**  | **Units**  | **Description** |
|------------------|----------|-----------|----------------|
| `repo`          | TEXT     | -         | Name of the repository. |
| `owner`         | TEXT     | -         | GitHub username or organization that owns the repository. |
| `release_no`    | INTEGER  | -         | Sequential release number of the project. |
| `release_date`  | DATE     | YYYY-MM-DD | Date when the release was published. |
| `size`         | INTEGER  | **Lines of Code (LOC)** | Size of the release in terms of source code. |

---

## 📌 Table: `commit_stats`
This table contains developer activity data related to commits.

| **Field**             | **Type**   | **Units**  | **Description** |
|----------------------|----------|-----------|----------------|
| `commit_id`         | TEXT     | -         | Unique identifier (hash) for the commit. |
| `dev_id`           | TEXT     | -         | Developer identifier (GitHub user ID). |
| `repo`            | TEXT     | -         | Name of the repository. |
| `owner`           | TEXT     | -         | Repository owner. |
| `dev_time`        | FLOAT    | **Hours** | Time spent by the developer on the commit. |
| `sloc_modifications` | INTEGER  | **Lines of Code (LOC)** | Number of source lines of code modified in the commit. |
| `productivity_index` | FLOAT    | **0-1 Scale** | Index representing developer productivity. |
| `skill_factor`    | FLOAT    | **0-1 Scale** | Skill factor based on historical commits. |

---

## 📌 Table: `release_effort_estimate`
This table contains release-level effort estimation metrics.

| **Field**            | **Type**  | **Units**  | **Description** |
|---------------------|----------|-----------|----------------|
| `repo`             | TEXT     | -         | Name of the repository. |
| `owner`            | TEXT     | -         | Repository owner. |
| `min_release_id`   | INTEGER  | -         | ID of the earliest considered release. |
| `max_release_id`   | INTEGER  | -         | ID of the latest considered release. |
| `start_release_date` | DATE     | YYYY-MM-DD | Start date of the release period. |
| `end_release_date` | DATE     | YYYY-MM-DD | End date of the release period. |
| `days`            | INTEGER  | **Days**  | Number of days between start and end release dates. |
| `dev_count`       | INTEGER  | **Count** | Number of developers contributing to the release. |
| `effort`          | FLOAT    | **Person-Months** | Estimated effort required for development, calculated as the total person-months worked. |

---

## 📌 Table: `soft_desc_pva_vec`
This table contains **vectorized software descriptions** and **similarity scores**.

| **Field**       | **Type**  | **Units**  | **Description** |
|---------------|----------|-----------|----------------|
| `category`    | TEXT     | -         | Software category (e.g., library, framework). |
| `owner`       | TEXT     | -         | Repository owner. |
| `repo`        | TEXT     | -         | Repository name. |
| `pv_vec`      | BLOB/TEXT | -         | Precomputed vector representation of software description. |
| `cos_sim`     | FLOAT    | **Cosine Similarity (0-1)** | Measures textual similarity between projects (1 = identical, 0 = completely different). |

---

## 📌 Table: `avg_repo_effort`
This table contains **aggregated effort estimates** for repositories.

| **Field**      | **Type**  | **Units**  | **Description** |
|--------------|----------|-----------|----------------|
| `owner`      | TEXT     | -         | Repository owner. |
| `repo`       | TEXT     | -         | Repository name. |
| `devCount`   | INTEGER  | **Count** | Number of developers involved in the project. |
| `activDays`  | INTEGER  | **Days**  | Total number of active development days (not necessarily consecutive). |
| `totlDays`   | INTEGER  | **Days**  | Total project duration since inception. |
| `modLOC`     | INTEGER  | **Lines of Code (LOC)** | Number of modified source lines of code. |
| `daf`        | FLOAT    | **0-1 Scale** | Developer Activity Factor (higher values indicate more activity). |
| `effort`     | FLOAT    | **Person-Months** | Estimated total development effort in **person-months** (1 person working full-time for a month). |

---

# **📌 Summary of Units in the Dataset**
- **Time-Based Fields**:
  - `dev_time` (Hours)
  - `activDays`, `totlDays`, `days` (Days)
  - `effort` (**Person-Months**)

- **Code-Based Fields**:
  - `size`, `sloc_modifications`, `modLOC` (Lines of Code)

- **Similarity-Based Fields**:
  - `cos_sim` (Cosine Similarity, **0-1** scale)

- **Productivity & Activity Metrics**:
  - `productivity_index`, `skill_factor`, `daf` (**0-1 scales**)

---

