import sqlite3
import requests
import os
import pandas as pd
import getpass
import time
from datetime import datetime

# GitHub API Base URLs
GITHUB_REST_API_URL = "https://api.github.com/repos"
GITHUB_GRAPHQL_API_URL = "https://api.github.com/graphql"

github_token = getpass.getpass("Enter your GitHub Personal Access Token: ")
headers = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'Bearer {github_token}'
}

def check_rate_limit():
    response = requests.get("https://api.github.com/rate_limit", headers=headers)
    if response.status_code == 200:
        rate_data = response.json()
        remaining = rate_data["rate"]["remaining"]
        reset_time = rate_data["rate"]["reset"]
        print(f"GitHub API Rate Limit Remaining: {remaining}, Resets at: {time.ctime(reset_time)}")
        return remaining
    else:
        print("Failed to fetch rate limit info.")
        return 0

def fetch_repo_metadata(owner, repo):
    url = f"{GITHUB_REST_API_URL}/{owner}/{repo}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("created_at"), data.get("size", 0), data.get("description", "")
    return None, 0, ""
    return None, 0

def fetch_developer_metrics(owner, repo):
    contributors_url = f"{GITHUB_REST_API_URL}/{owner}/{repo}/contributors?anon=true"
    commits_url = f"{GITHUB_REST_API_URL}/{owner}/{repo}/commits"
    stats_url = f"{GITHUB_REST_API_URL}/{owner}/{repo}/stats/code_frequency"
    releases_url = f"{GITHUB_REST_API_URL}/{owner}/{repo}/releases"

    developer_count = 0
    time_range_days = 0
    sloc_added = 0
    sloc_deleted = 0
    sloc_modified = 0
    commit_count = 0
    first_commit_date = None
    first_release_date = None
    days_to_first_release = None

    try:
        contrib_response = requests.get(contributors_url, headers=headers)
        if contrib_response.status_code == 200:
            developers = contrib_response.json()
            developer_count = len(developers)

        commit_response = requests.get(commits_url, headers=headers, params={"per_page": 100})
        if commit_response.status_code == 200:
            commits = commit_response.json()
            commit_count = len(commits)
            if len(commits) >= 2:
                first_commit_date = datetime.strptime(commits[-1]['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
                last_commit_date = datetime.strptime(commits[0]['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
                time_range_days = (last_commit_date - first_commit_date).days

        for _ in range(5):
            stats_response = requests.get(stats_url, headers=headers)
            if stats_response.status_code == 202:
                time.sleep(2)
                continue
            elif stats_response.status_code == 200:
                stats = stats_response.json()
                for week in stats:
                    sloc_added += week[1]
                    sloc_deleted += abs(week[2])
                sloc_modified = sloc_added + sloc_deleted
                break

        release_response = requests.get(releases_url, headers=headers)
        if release_response.status_code == 200:
            releases = release_response.json()
            if releases:
                first_release_date = datetime.strptime(releases[-1]['created_at'], "%Y-%m-%dT%H:%M:%SZ")

        if first_commit_date and first_release_date:
            delta = (first_release_date - first_commit_date).days
            days_to_first_release = max(0, delta)

    except Exception as e:
        print(f"Error fetching metrics for {owner}/{repo}: {e}")

    effort = developer_count * time_range_days
    effort_months = round(effort / 30.0, 2)
    return developer_count, time_range_days, effort, effort_months, sloc_added, sloc_deleted, sloc_modified, days_to_first_release, first_commit_date, first_release_date, commit_count

def process_repositories(cursor, repositories, limit=None):
    if limit:
        repositories = repositories[:limit]
    batch_size = 20
    all_records = []
    for i in range(0, len(repositories), batch_size):
        check_rate_limit()
        batch = repositories[i:i + batch_size]
        for owner, repo in batch:
            created_at, size, description = fetch_repo_metadata(owner, repo)
            dev_count, days_active, effort, effort_months, sloc_add, sloc_del, sloc_mod, days_to_release, first_commit, first_release, commit_count = fetch_developer_metrics(owner, repo)
            row = (
                owner, repo, description,
                str(created_at) if created_at else None,
                str(first_commit) if first_commit else None,
                str(first_release) if first_release else None,
                commit_count, 0, size, dev_count, days_active,
                effort, effort_months, sloc_add, sloc_del, sloc_mod, days_to_release
            )
            cursor.execute('''
                INSERT INTO repo_additional_info (
                    owner, repo, description, created_at, first_commit_date, first_release_date, commit_count,
                    num_files_dirs, size_kb, developer_count, development_days,
                    effort_score, effort_months, sloc_added, sloc_deleted,
                    sloc_modified, days_to_first_release
                ) VALUES (?, ?, ?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)
            all_records.append(row)

    df = pd.DataFrame(all_records, columns=[
        "owner", "repo", "description", "created_at", "first_commit_date", "first_release_date", "commit_count",
        "num_files_dirs", "size_kb", "developer_count", "development_days",
        "effort_score", "effort_months", "sloc_added", "sloc_deleted",
        "sloc_modified", "days_to_first_release"])
    df.to_csv("sdee_lite_description.csv", index=False)

def main(mode="all"):
    clean_sql_path = 'sdee_lite_cleaned.sql'
    new_db_path = 'sdee_lite_description.db'
    new_sql_path = 'sdee_lite_description.sql'

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    with open(clean_sql_path, "r", encoding="utf-8") as file:
        sql_script = file.read()
    cursor.executescript(sql_script)

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS repo_additional_info (
            owner TEXT,
            repo TEXT,            
            description TEXT,
            created_at TEXT,
            first_commit_date TEXT,
            first_release_date TEXT,
            commit_count INTEGER,
            num_files_dirs INTEGER,
            size_kb INTEGER,
            developer_count INTEGER,
            development_days INTEGER,
            effort_score INTEGER,
            effort_months REAL,
            sloc_added INTEGER,
            sloc_deleted INTEGER,
            sloc_modified INTEGER,
            days_to_first_release INTEGER
        )
    ''')

    cursor.execute('SELECT owner, repo FROM repo_info_pv_vec')
    repositories = cursor.fetchall()

    if mode == "test":
        process_repositories(cursor, repositories, limit=20)
    else:
        process_repositories(cursor, repositories)

    new_conn = sqlite3.connect(new_db_path)
    with new_conn:
        new_cursor = new_conn.cursor()
        new_cursor.execute('''
            CREATE TABLE IF NOT EXISTS repo_additional_info (
                owner TEXT,
                repo TEXT,
                description TEXT,
                created_at TEXT,
                first_commit_date TEXT,
                first_release_date TEXT,
                commit_count INTEGER,
                num_files_dirs INTEGER,
                size_kb INTEGER,
                developer_count INTEGER,
                development_days INTEGER,
                effort_score INTEGER,
                effort_months REAL,
                sloc_added INTEGER,
                sloc_deleted INTEGER,
                sloc_modified INTEGER,
                days_to_first_release INTEGER
            )
        ''')
        cursor.execute("SELECT * FROM repo_additional_info")
        data = cursor.fetchall()
        new_cursor.executemany('''
            INSERT INTO repo_additional_info (
                owner, repo, description, created_at, first_commit_date, first_release_date, commit_count,
                num_files_dirs, size_kb, developer_count, development_days,
                effort_score, effort_months, sloc_added, sloc_deleted,
                sloc_modified, days_to_first_release
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
    new_conn.commit()
    new_conn.close()

    with open(new_sql_path, "w", encoding="utf-8") as file:
        for line in conn.iterdump():
            file.write(f"{line}\n")

    conn.close()
    print(f"Database saved as {new_db_path}, SQL script as {new_sql_path}, and CSV as sdee_lite_description.csv")

if __name__ == '__main__':
    # mode = input("Enter mode ('test' for 20 records, 'all' for all records): ").strip().lower()
    mode = 'test'
    # mode = 'all'
    main(mode)
