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