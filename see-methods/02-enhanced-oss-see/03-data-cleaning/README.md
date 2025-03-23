# 🧹 OSS Effort Estimation - Data Cleaning Pipeline

This document outlines all the data cleaning stages applied to the `repo_additional_info` dataset before estimating software development effort. The cleaning process ensures only meaningful, high-quality, English textual data is preserved.

---

## ✅ Overview

- **Input Source**: `sdee_lite_description.sql`
- **Target Table**: `repo_additional_info`
- **Output Files**:
  - Cleaned SQL Dump: `sdee_lite_description_cleaned.sql`
  - Cleaning Log: `removed-records.log`

---

## 🔄 Cleaning Pipeline Stages

### 1. **Load & Normalize Input**
- Load the SQLite dump into memory.
- Ensure `description` is treated as a string.
- Strip whitespace.

---

### 2. **Remove Records with `description = 'None'`**
- Case-insensitive match for:
  - `"None"`
  - Python `None`
- These are meaningless placeholders.

---

### 3. **Remove `description` that is `"null"` or variations**
- Matches:
  - `"NULL"`
  - `"null"`
  - `[null]`, `[NULL]`, etc.
- Converted to `NaN` and dropped later as null.

---

### 4. **Remove Descriptions Exceeding 3500 Characters**
- Such descriptions are unusually long and likely contain non-relevant or auto-generated content.

---

### 5. **Remove Records with Forbidden Terms**
- If `description` contains:
  - `[DEPRECATED]`, `READ-ONLY`, `THIS REPO IS UNMAINTAINED`, `UNMAINTAINED`, `ABANDONED`, etc.
- Regex is case-insensitive and removes surrounding brackets.
- Statistics are tracked **before and after** cleaning.

---

### 6. **Remove Records with Specific "Unmaintained" Text**
- Specifically: `"This project is not maintained anymore"` (case-insensitive)

---

### 7. **Remove Records with Null or Empty Descriptions**
- Either:
  - Missing (`NaN`)
  - Empty string after stripping
  - Whitespace only

---

### 8. **Remove Emojis and Special Symbols**
- All Unicode emoji blocks are removed using a regex.
- Covers:
  - Flags
  - Symbols
  - Emoticons
  - Dingbats (e.g., `✅`, `✨`, `📦`, `🚀`, etc.)

---

### 9. **Remove Non-English Words**
- Keeps only words matching `[a-zA-Z]{2,}`
- Removes any bracketed content (e.g., Chinese/Japanese inside brackets).
- Example cleaned:  
  `"中文自然语言处理 Toolkit for NLP"` → `"Toolkit for NLP"`

---

### 10. **Normalize Unicode**
- Converts all descriptions to a canonical Unicode format (`NFKC`).
- Prevents issues caused by invisible, composite, or unusual characters.

---

### 11. **Synchronize Removals Across Tables**
When a record is removed from `repo_additional_info`, it is also deleted from all tables where:
- Both `owner` and `repo` match
- Affected tables:
  - `avg_repo_effort`
  - `features`
  - `release_info`
  - `release_wise_effort`
  - `repo_info_pv_vec`

---

### 12. **Update Cleaned Descriptions into SQLite**
- All remaining (non-removed) cleaned descriptions are written back to the `repo_additional_info` table in the SQLite database.

---

## 📤 Output Files

### `sdee_lite_description_cleaned.sql`
- Cleaned in-memory SQLite database exported to disk.

### `removed-records.log`
- JSON log containing:
  - Per-condition removal statistics
  - List of all removed `owner` + `repo` pairs

---

## 📊 Tracked Removal Categories

- `too_long`: Description too long
- `forbidden_words`: Forbidden/deprecated/abandoned phrases
- `unmaintained_text`: "Not maintained anymore"
- `null_or_empty`: Empty or missing descriptions
- `none_text`: Description was "None"
- `total_removed`: Total number of rows removed

---

## ✅ Notes
- Descriptions are cleaned **before any downstream vectorization or effort estimation**.
- This ensures better embeddings, model performance, and effort prediction accuracy.

