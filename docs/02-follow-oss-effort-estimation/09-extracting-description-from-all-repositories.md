# Overview of `crawler-for-all-repositories.py`

This document describes the functionality of the `crawler-for-all-repositories.py` script, which fetches repository details from GitHub using both the REST and GraphQL APIs and stores the information in an SQLite database.

## 1. Script Components

### 1.1 Imports
The script imports several Python modules for handling API requests, database operations, authentication, and time-related functions:
- `sqlite3` for interacting with an SQLite database.
- `requests` for making HTTP requests to the GitHub API.
- `os` for handling file paths.
- `pandas` for working with data in tabular format.
- `getpass` for securely retrieving the GitHub token.
- `time` for handling API rate limits and time conversions.

### 1.2 GitHub API Configuration
- The script defines the base URLs for the GitHub REST and GraphQL APIs.
- It prompts the user to enter a GitHub personal access token (`github_token`) for authentication.
- Headers for API requests include the authentication token and the appropriate `Accept` header for GitHub API versioning.

## 2. Functions

### 2.1 `check_rate_limit()`
- Checks the remaining API rate limits using GitHub’s `/rate_limit` endpoint.
- Displays the number of remaining requests and the reset time.
- Returns the number of remaining requests.

### 2.2 `fetch_repos_graphql(repos, batch_size)`
- Fetches repository details using the GitHub GraphQL API.
- Queries repository names, owners, descriptions, disk usage, and primary programming languages.
- Handles API rate limits by reducing batch size and retrying if needed.
- Returns a dictionary with repository data.

### 2.3 `process_repositories(cursor, repositories, limit=None)`
- Processes repositories by retrieving their details using `fetch_repos_graphql()`.
- Inserts fetched repository metadata (owner, repo name, description, size, and languages) into an SQLite database.
- Limits the number of processed repositories if specified.

### 2.4 `display_table_samples(conn)`
- Retrieves and prints the first 5 rows from each table in the SQLite database for validation.

## 3. Main Execution (`main(mode)`)

### 3.1 Database Setup
- Defines paths for the cleaned SQL database and a new database with descriptions.
- Loads an SQLite database schema from an external SQL file (`sdee_lite_cleaned.sql`).
- Creates a new table `repo_additional_info` to store repository metadata.

### 3.2 Fetching and Storing Repository Data
- Queries repository names and owners from an existing table (`repo_info_pv_vec`).
- Calls `process_repositories()` to fetch repository details and store them in SQLite.
- Saves the updated database to `sdee-with-description/sdee_lite_description.db`.
- Exports the new dataset schema to a SQL script (`sdee_lite_description.sql`).

### 3.3 Displaying Data
- Calls `display_table_samples()` to show sample repository records from the SQLite database.

## 4. Script Execution
The script runs in either:
- **Test mode** (`mode="test"`) to process only 20 repositories.
- **Full mode** (`mode="all"`) to process all repositories.

The script execution starts with:
```python
if __name__ == '__main__':
    mode = "all"
    main(mode)
```

## 5. Summary
1. Connects to an SQLite database and loads an existing schema.
2. Fetches repository metadata from GitHub via GraphQL.
3. Stores the fetched metadata in a new table.
4. Saves the updated database and exports an SQL script.
5. Displays sample data for validation.



