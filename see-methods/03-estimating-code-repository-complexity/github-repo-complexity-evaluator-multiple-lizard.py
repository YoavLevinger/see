# GitHub Code Complexity Evaluator
# This script clones a GitHub repo, extracts code and git metrics,
# computes a normalized complexity score, prints, and exports to CSV.

import os
import subprocess
import shutil
import tempfile
import pandas as pd
import lizard  # Multi-language static analysis
from pydriller import Repository
from pprint import pprint
import matplotlib.pyplot as plt

# Helper function to clone a GitHub repository
def clone_repo(owner, repo_name):
    temp_dir = tempfile.mkdtemp()
    repo_url = f"https://github.com/{owner}/{repo_name}.git"
    repo_path = os.path.join(temp_dir, repo_name)
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    return repo_path, temp_dir

# Analyze code complexity using Lizard (multi-language support)
def analyze_code_metrics(repo_path):
    code_metrics = {
        "loc": 0,
        "cyclomatic_complexity": 0,
        "halstead_volume": 0,  # Placeholder; Halstead volume is approximated
        "function_count": 0
    }
    # Run Lizard to analyze the entire repo recursively
    analysis = lizard.analyze([repo_path])
    for file in analysis:
        code_metrics["loc"] += file.nloc
        for function in file.function_list:
            code_metrics["cyclomatic_complexity"] += function.cyclomatic_complexity
            code_metrics["function_count"] += 1
            # Approximate Halstead Volume using token count * average complexity (simple proxy)
            code_metrics["halstead_volume"] += function.token_count * 1.5
    return code_metrics

# Analyze Git history metrics using PyDriller
def analyze_git_metrics(repo_path):
    git_metrics = {
        "commits": 0,
        "contributors": set(),
        "lines_added": 0,
        "lines_deleted": 0
    }
    for commit in Repository(repo_path).traverse_commits():
        git_metrics["commits"] += 1
        git_metrics["contributors"].add(commit.author.email)
        git_metrics["lines_added"] += commit.insertions
        git_metrics["lines_deleted"] += commit.deletions
    git_metrics["contributors"] = len(git_metrics["contributors"])
    return git_metrics

# Normalize complexity metrics using preset maximums
def normalize_metrics(metrics, max_values):
    normalized = {}
    for key in metrics:
        normalized[key] = metrics[key] / max_values[key] if max_values[key] != 0 else 0
    return normalized

# Evaluate final complexity score based on normalized values
def evaluate_complexity(normalized_metrics):
    score = (0.3 * normalized_metrics["loc"] +
             0.4 * normalized_metrics["cyclomatic_complexity"] +
             0.3 * normalized_metrics["halstead_volume"])
    return round(score, 4)

# Main function to orchestrate the workflow for a single repo
def evaluate_repo_complexity(owner, repo_name):
    repo_path, temp_dir = clone_repo(owner, repo_name)
    code_metrics = analyze_code_metrics(repo_path)
    git_metrics = analyze_git_metrics(repo_path)
    all_metrics = {**code_metrics, **git_metrics}

    max_values = {
        "loc": 50000,
        "cyclomatic_complexity": 2000,
        "halstead_volume": 100000
    }
    normalized = normalize_metrics({
        "loc": code_metrics["loc"],
        "cyclomatic_complexity": code_metrics["cyclomatic_complexity"],
        "halstead_volume": code_metrics["halstead_volume"]
    }, max_values)

    complexity_score = evaluate_complexity(normalized)

    all_metrics["owner"] = owner
    all_metrics["repo"] = repo_name
    all_metrics["complexity_score"] = complexity_score

    shutil.rmtree(temp_dir)
    return all_metrics

# Batch evaluation of multiple repositories
def evaluate_multiple_repos(repo_list, output_csv="batch_repo_complexity.csv"):
    all_results = []
    for owner, repo in repo_list:
        print(f"\nEvaluating {owner}/{repo}...")
        try:
            metrics = evaluate_repo_complexity(owner, repo)
            all_results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {owner}/{repo}: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nBatch results saved to: {output_csv}")

    # Create a simple bar chart dashboard
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by="complexity_score", ascending=False)
    plt.barh(df_sorted["repo"], df_sorted["complexity_score"])
    plt.xlabel("Complexity Score")
    plt.title("Repository Complexity Scores")
    plt.tight_layout()
    plt.savefig("complexity_dashboard.png")
    print("Dashboard saved as: complexity_dashboard.png")
    plt.show()


# Example usage: evaluate a batch of GitHub repositories
repo_list = [('neva-dev','javarel-osgi-plugin'), ('AndyObtiva','DatingNetwork'), ('lovell','sharp')]
evaluate_multiple_repos(repo_list)

