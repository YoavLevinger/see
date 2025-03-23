import sqlite3
import requests
import os
import pandas as pd
import getpass
import time

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
        time.sleep(60)  # Wait before retrying

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
    query += "}"  # Closing the GraphQL query

    response = requests.post(GITHUB_GRAPHQL_API_URL, json={"query": query}, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", {})
    elif response.status_code == 400 and batch_size > 5:
        print("GraphQL query too large, reducing batch size and retrying...")
        return fetch_repos_graphql(repos[:batch_size // 2], batch_size // 2)  # Reduce batch size and retry
    else:
        print(f"GraphQL API Error: {response.status_code}")
        return {}


# Function to process repositories and insert data into the database
def process_repositories(cursor, repositories, limit=None):
    if limit:
        repositories = repositories[:limit]  # Limit to first N records if specified
    batch_size = 20  # Start with batch size of 20
    for i in range(0, len(repositories), batch_size):
        batch = repositories[i:i + batch_size]
        repo_data = fetch_repos_graphql(batch, batch_size)

        for key in repo_data:
            repo_info = repo_data[key]
            if repo_info:
                owner = repo_info['owner']['login']
                repo = repo_info['name']
                description = repo_info.get('description', 'No description')
                size = repo_info.get('diskUsage', 0)  # Size in KB
                languages = [lang['name'] for lang in repo_info.get('languages', {}).get('nodes', [])]
                languages_str = ', '.join(languages) if languages else 'Not specified'

                cursor.execute('''
                    INSERT INTO repo_additional_info (owner, repo, description, num_files_dirs, languages, size_kb)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (owner, repo, description, 0, languages_str, size))


# Function to query each table and display the first 5 rows
def display_table_samples(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        print(f"\nFirst 5 rows of table: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
        print(df)


# Main function to execute the script
def main(mode="all"):
    # Define database paths
    clean_sql_path = 'sdee_lite_cleaned.sql'
    new_db_path = 'sdee-with-description/sdee_lite_description.db'
    new_sql_path = '../02-dataset/sdee_mysql_description.sql'

    # Connect to SQLite (in-memory database)
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Read and execute the cleaned SQL script
    with open(clean_sql_path, "r", encoding="utf-8") as file:
        sql_script = file.read()
    cursor.executescript(sql_script)

    print("Database loaded successfully!")

    # Create the new table in-memory
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS repo_additional_info (
            owner TEXT,
            repo TEXT,
            description TEXT,
            num_files_dirs INTEGER,
            languages TEXT,
            size_kb INTEGER
        )
    ''')

    # Query to retrieve owner and repo from the existing table
    cursor.execute('SELECT owner, repo FROM repo_info_pv_vec')
    repositories = cursor.fetchall()

    # Process repositories using GraphQL
    if mode == "test":
        process_repositories(cursor, repositories, limit=20)  # Fetch only 20 records
    else:
        process_repositories(cursor, repositories)  # Fetch all records

    # Save to new database file
    new_conn = sqlite3.connect(new_db_path)
    with new_conn:
        new_cursor = new_conn.cursor()
        new_cursor.executescript(sql_script)  # Ensure schema exists
        new_cursor.execute('''
            CREATE TABLE IF NOT EXISTS repo_additional_info (
                owner TEXT,
                repo TEXT,
                description TEXT,
                num_files_dirs INTEGER,
                languages TEXT,
                size_kb INTEGER
            )
        ''')
        cursor.execute("SELECT * FROM repo_additional_info")
        data = cursor.fetchall()
        new_cursor.executemany('''
            INSERT INTO repo_additional_info (owner, repo, description, num_files_dirs, languages, size_kb)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data)
    new_conn.commit()
    new_conn.close()

    # Save SQL script
    with open(new_sql_path, "w", encoding="utf-8") as file:
        for line in conn.iterdump():
            file.write(f"{line}\n")

    # Display sample data from all tables
    display_table_samples(conn)

    conn.close()
    print(f"Database saved as {new_db_path} and SQL script saved as {new_sql_path}")


if __name__ == '__main__':
    # mode = input("Enter mode ('test' for 20 records, 'all' for all records): ").strip().lower()
    mode = "test"
    # mode = "all"
    main(mode)
