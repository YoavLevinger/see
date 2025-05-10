import sqlite3
import pandas as pd
import numpy as np
import re
import json
from tabulate import tabulate
import unicodedata


def load_data(sql_file="sdee_lite_description.sql"):
    print(f"📂 Loading data from {sql_file} into memory...")
    conn = sqlite3.connect(":memory:")
    with open(sql_file, "r", encoding="utf-8") as file:
        sql_script = file.read()
    conn.executescript(sql_script)
    df = pd.read_sql_query("SELECT * FROM repo_additional_info;", conn)
    print("✅ Data loaded into memory successfully!")
    return df, conn


def get_table_counts(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    return {table: cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}


def normalize_text(text):
    return unicodedata.normalize("NFKC", text)


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F1E0-\U0001F1FF"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F600-\U0001F64F"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FAFF"
        u"\U00002600-\U000026FF"
        u"\U00002700-\U000027BF"
        u"\U00002B50-\U00002B59"
        u"\U00002500-\U00002BEF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)


def remove_non_english_words(text):
    text = re.sub(r'[\(\[【{<].*?[\)】\]>}]', '', text)
    return " ".join(re.findall(r'\b[a-zA-Z]{2,}\b', text))


def clean_data(df, conn, sql_save_path="sdee_lite_description_cleaned.sql", log_file="removed-records.log"):
    print("🧹 Cleaning data...")
    df = df.copy()
    initial_counts = get_table_counts(conn)
    removal_stats = {
        "too_long": 0, "forbidden_words": 0, "unmaintained_text": 0,
        "null_or_empty": 0, "none_text": 0, "total_removed": 0
    }
    to_remove = pd.Series(False, index=df.index)

    # STEP 1: Normalize and remove 'None' (as string or value)
    df["description"] = df["description"].astype(str).str.strip()
    is_none = df["description"].str.lower() == "none"
    removal_stats["none_text"] = is_none.sum()
    to_remove |= is_none

    # Also remove exact NULL strings and null values
    df["description"] = df["description"].replace(
        r"(?i)^(\[)?null(\])?$", np.nan, regex=True
    )

    # STEP 2: Remove too-long descriptions
    too_long = df["description"].str.len() > 3500
    to_remove |= too_long
    removal_stats["too_long"] = too_long.sum()

    # STEP 3: Remove forbidden words and brackets
    forbidden_words = ["deprecated", "read-only", "read only", "this repo is unmaintained", "unmaintained", "abandoned"]
    pattern = r"(?i)[\[\(]?\s*(" + "|".join(forbidden_words) + r")\s*[\]\)]?"
    df["description"] = df["description"].str.replace(pattern, "", regex=True)
    df["description"] = df["description"].str.replace(r"[\[\]\(\){}]", "", regex=True)

    # Count forbidden words post-cleanup for stats
    removal_stats["forbidden_words"] = df["description"].str.contains(
        "|".join(forbidden_words), case=False, na=False).sum()

    # STEP 4: Remove unmaintained repos
    unmaintained = df["description"].str.contains("not maintained anymore", case=False, na=False)
    to_remove |= unmaintained
    removal_stats["unmaintained_text"] = unmaintained.sum()

    # STEP 5: Remove null or empty descriptions
    null_or_empty = df["description"].isnull() | (df["description"].str.strip() == "")
    to_remove |= null_or_empty
    removal_stats["null_or_empty"] = null_or_empty.sum()

    # STEP 6: Clean descriptions
    df.loc[~to_remove, "description"] = df.loc[~to_remove, "description"].apply(
        lambda x: remove_emojis(x) if isinstance(x, str) else x
    )
    df.loc[~to_remove, "description"] = df.loc[~to_remove, "description"].apply(
        lambda x: remove_non_english_words(x) if isinstance(x, str) else x
    )
    df.loc[~to_remove, "description"] = df.loc[~to_remove, "description"].apply(
        lambda x: normalize_text(x) if isinstance(x, str) else x
    )

    # STEP 7: Record and remove
    removed = df.loc[to_remove]
    df = df.loc[~to_remove]
    removal_stats["total_removed"] = len(removed)

    # STEP 8: Remove matching owner+repo in ALL relevant tables
    affected = removed[["owner", "repo"]] if not removed.empty else pd.DataFrame(columns=["owner", "repo"])
    tables = ["avg_repo_effort", "features", "release_info", "release_wise_effort", "repo_additional_info", "repo_info_pv_vec"]
    cursor = conn.cursor()
    for table in tables:
        if table in initial_counts:
            for _, row in affected.iterrows():
                cursor.execute(f"DELETE FROM {table} WHERE owner = ? AND repo = ?", (row["owner"], row["repo"]))
    conn.commit()

    # ✅ STEP 9: UPDATE cleaned descriptions into SQLite
    for _, row in df.iterrows():
        cursor.execute(
            "UPDATE repo_additional_info SET description = ? WHERE owner = ? AND repo = ?",
            (row["description"], row["owner"], row["repo"])
        )
    conn.commit()

    # 💾 STEP 10: Save SQL dump
    with open(sql_save_path, "w", encoding="utf-8") as f:
        for line in conn.iterdump():
            f.write(f"{line}\n")

    # 📝 Save removal log
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "removal_stats": {k: int(v) for k, v in removal_stats.items()},
            "removed_records": removed.to_dict(orient="records")
        }, f, indent=4)

    # 📊 Final stats
    final_counts = get_table_counts(conn)
    print("\n📊 Removal Statistics:")
    for k, v in removal_stats.items():
        print(f"  {k.replace('_', ' ').title()}: {v} records removed")

    print("\n📋 Table Record Counts Before & After Cleaning:")
    rows = [[t, initial_counts.get(t, 0), final_counts.get(t, 0), initial_counts.get(t, 0) - final_counts.get(t, 0)]
            for t in initial_counts]
    print(tabulate(rows, headers=["Table", "Before", "After", "Removed"], tablefmt="pretty"))


def main():
    print("🚀 Starting OSS Effort Estimation Cleaning...")
    df, conn = load_data()
    clean_data(df, conn)
    print("\n✅ Cleaning complete. Cleaned DB saved as 'sdee_lite_description_cleaned.sql'")


if __name__ == "__main__":
    main()
