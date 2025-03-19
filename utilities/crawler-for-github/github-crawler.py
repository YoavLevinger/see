import requests


def get_github_repo_description(owner, repo):
    """
    Fetches the repository description and README content from a GitHub repository.
    """
    base_url = "https://api.github.com/repos/{}/{}".format(owner, repo)
    headers = {"Accept": "application/vnd.github.v3+json"}

    # Get repository details
    repo_response = requests.get(base_url, headers=headers)
    if repo_response.status_code != 200:
        return f"Error: Unable to fetch repository details (Status Code: {repo_response.status_code})"
    repo_data = repo_response.json()
    description = repo_data.get("description", "No description available.")

    # Get README file
    readme_url = f"{base_url}/readme"
    readme_response = requests.get(readme_url, headers=headers)
    readme_content = ""

    if readme_response.status_code == 200:
        readme_data = readme_response.json()
        readme_content = requests.get(readme_data.get("download_url", "")).text
    else:
        readme_content = "README not found."

    return {
        "Repository Description": description,
        "README Content": readme_content[:1000]  # Limiting to first 1000 chars for brevity
    }


if __name__ == "__main__":
    owner = "juxt" #input("Enter GitHub repository owner: ")
    repo = "aero" #input("Enter repository name: ")
    details = get_github_repo_description(owner, repo)
    print("\nRepository Description:")
    print(details["Repository Description"])
    print("\nREADME Content (First 1000 chars):")
    print(details["README Content"])
