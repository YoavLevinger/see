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

# Get GitHub Token Securely
github_token = getpass.getpass("Enter your GitHub Personal Access Token: ")
headers = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'Bearer {github_token}'
}


# Function to check GitHub API rate limits
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


# Function to fetch repository details using GraphQL for batch queries
def fetch_repos_graphql(repos, batch_size):
    if check_rate_limit() == 0:
        print("Rate limit exceeded, waiting for reset...")
        time.sleep(60)

    query = """
    {
    """
    for i, (owner, repo) in enumerate(repos):
        query += f"""
        repo{i}: repository(owner: \"{owner}\", name: \"{repo}\") {{
            name
            owner {{ login }}
            description
            diskUsage
            languages(first: 5) {{ nodes {{ name }} }}
        }}
        """
    query += "}"

    response = requests.post(GITHUB_GRAPHQL_API_URL, json={"query": query}, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", {})
    elif response.status_code == 400 and batch_size > 5:
        print("GraphQL query too large, reducing batch size and retrying...")
        return fetch_repos_graphql(repos[:batch_size // 2], batch_size // 2)
    else:
        print(f"GraphQL API Error: {response.status_code}")
        return {}


# Function to fetch developer activity metrics using REST API
def fetch_developer_metrics(owner, repo):
    contributors_url = f"{GITHUB_REST_API_URL}/{owner}/{repo}/contributors?anon=true"
    commits_url = f"{GITHUB_REST_API_URL}/{owner}/{repo}/commits"
    stats_url = f"{GITHUB_REST_API_URL}/{owner}/{repo}/stats/code_frequency"

    developer_count = 0
    time_range_days = 0
    sloc_added = 0
    sloc_deleted = 0
    sloc_modified = 0

    try:
        contrib_response = requests.get(contributors_url, headers=headers)
        if contrib_response.status_code == 200:
            developers = contrib_response.json()
            developer_count = len(developers)

        commit_response = requests.get(commits_url, headers=headers, params={"per_page": 100})
        if commit_response.status_code == 200:
            commits = commit_response.json()
            if len(commits) >= 2:
                last_date = datetime.strptime(commits[0]['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
                first_date = datetime.strptime(commits[-1]['commit']['committer']['date'], "%Y-%m-%dT%H:%M:%SZ")
                time_range_days = (last_date - first_date).days

        stats_response = requests.get(stats_url, headers=headers)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            for week in stats:
                sloc_added += week[1]
                sloc_deleted += abs(week[2])
            sloc_modified = sloc_added + sloc_deleted

    except Exception as e:
        print(f"Error fetching developer metrics for {owner}/{repo}: {e}")

    effort = developer_count * time_range_days
    effort_months = round(effort / 30.0, 2)
    return developer_count, time_range_days, effort, effort_months, sloc_added, sloc_deleted, sloc_modified


# Function to process repositories and insert data into the database
def process_repositories(cursor, repositories, limit=None):
    if limit:
        repositories = repositories[:limit]
    batch_size = 20
    for i in range(0, len(repositories), batch_size):
        batch = repositories[i:i + batch_size]
        repo_data = fetch_repos_graphql(batch, batch_size)

        for key in repo_data:
            repo_info = repo_data[key]
            if repo_info:
                owner = repo_info['owner']['login']
                repo = repo_info['name']
                description = repo_info.get('description', 'No description')
                size = repo_info.get('diskUsage', 0)
                languages = [lang['name'] for lang in repo_info.get('languages', {}).get('nodes', [])]
                languages_str = ', '.join(languages) if languages else 'Not specified'

                dev_count, days_active, effort, effort_months, sloc_add, sloc_del, sloc_mod = fetch_developer_metrics(
                    owner, repo)

                cursor.execute('''
                    INSERT INTO repo_additional_info (
                        owner, repo, description, num_files_dirs, languages, size_kb,
                        developer_count, development_days, effort_score, effort_months,
                        sloc_added, sloc_deleted, sloc_modified
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (owner, repo, description, 0, languages_str, size, dev_count, days_active, effort, effort_months,
                      sloc_add, sloc_del, sloc_mod))


# Display samples
def display_table_samples(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        print(f"\nFirst 5 rows of table: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(df)


# Main logic
def main(mode="all"):
    clean_sql_path = 'sdee_lite_cleaned.sql'
    new_db_path = 'olds-runs/sdee_lite_description.db'
    new_sql_path = 'olds-runs/sdee_lite_description.sql'

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    with open(clean_sql_path, "r", encoding="utf-8") as file:
        sql_script = file.read()
    cursor.executescript(sql_script)
    print("Database loaded successfully!")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS repo_additional_info (
            owner TEXT,
            repo TEXT,
            description TEXT,
            num_files_dirs INTEGER,
            languages TEXT,
            size_kb INTEGER,
            developer_count INTEGER,
            development_days INTEGER,
            effort_score INTEGER,
            effort_months REAL,
            sloc_added INTEGER,
            sloc_deleted INTEGER,
            sloc_modified INTEGER
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
                num_files_dirs INTEGER,
                languages TEXT,
                size_kb INTEGER,
                developer_count INTEGER,
                development_days INTEGER,
                effort_score INTEGER,
                effort_months REAL,
                sloc_added INTEGER,
                sloc_deleted INTEGER,
                sloc_modified INTEGER
            )
        ''')
        cursor.execute("SELECT * FROM repo_additional_info")
        data = cursor.fetchall()
        new_cursor.executemany('''
            INSERT INTO repo_additional_info (
                owner, repo, description, num_files_dirs, languages, size_kb,
                developer_count, development_days, effort_score, effort_months,
                sloc_added, sloc_deleted, sloc_modified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
    new_conn.commit()
    new_conn.close()

    with open(new_sql_path, "w", encoding="utf-8") as file:
        for line in conn.iterdump():
            file.write(f"{line}\n")

    display_table_samples(conn)
    conn.close()
    print(f"Database saved as {new_db_path} and SQL script saved as {new_sql_path}")


if __name__ == '__main__':
    mode = input("Enter mode ('test' for 20 records, 'all' for all records): ").strip().lower()
    main(mode)