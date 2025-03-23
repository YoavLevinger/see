# GitHub Code Complexity Evaluator
# This script clones multiple GitHub repos, detects their primary language,
# analyzes supported ones using Lizard or JSON/YAML fallback, evaluates complexity,
# estimates development effort, prints metrics, saves to CSV, and generates a dashboard.

import os
import subprocess
import shutil
import tempfile
import pandas as pd
import lizard
from pydriller import Repository
from pprint import pprint
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
from contextlib import redirect_stdout
import io
import requests
import json
import yaml

# Helper function to clone a GitHub repository
def clone_repo(owner, repo_name):
    temp_dir = tempfile.mkdtemp()
    repo_url = f"https://github.com/{owner}/{repo_name}.git"
    repo_path = os.path.join(temp_dir, repo_name)
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    return repo_path, temp_dir

# Detect the main language of the GitHub repository using GitHub API
def detect_repo_language(owner, repo_name):
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data.get("language", "Unknown")
    except Exception:
        pass
    return "Unknown"

# Analyze JSON/YAML files for basic structure and nesting complexity
def analyze_json_yaml_complexity(repo_path):
    loc = 0
    max_nesting = 0

    def get_max_depth(data, current_depth=1):
        if isinstance(data, dict):
            return max((get_max_depth(v, current_depth + 1) for v in data.values()), default=current_depth)
        elif isinstance(data, list):
            return max((get_max_depth(v, current_depth + 1) for v in data), default=current_depth)
        else:
            return current_depth

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.json', '.yaml', '.yml')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                        loc += len(content.splitlines())
                        data = json.loads(content) if file.endswith('.json') else yaml.safe_load(content)
                        max_nesting = max(max_nesting, get_max_depth(data))
                except Exception:
                    continue

    return {
        "loc": loc,
        "cyclomatic_complexity": 0,
        "halstead_volume": max_nesting * 10,
        "function_count": 0
    }

# Analyze code complexity using Lizard
def analyze_code_metrics(repo_path):
    code_metrics = {"loc": 0, "cyclomatic_complexity": 0, "halstead_volume": 0, "function_count": 0}
    analysis = lizard.analyze([repo_path])
    for file in analysis:
        if not file.function_list:
            continue
        code_metrics["loc"] += file.nloc
        for function in file.function_list:
            code_metrics["cyclomatic_complexity"] += function.cyclomatic_complexity
            code_metrics["function_count"] += 1
            code_metrics["halstead_volume"] += function.token_count * 1.5
    return code_metrics

# Analyze Git metrics using PyDriller
def analyze_git_metrics(repo_path):
    git_metrics = {"commits": 0, "contributors": set(), "lines_added": 0, "lines_deleted": 0}
    for commit in Repository(repo_path).traverse_commits():
        git_metrics["commits"] += 1
        git_metrics["contributors"].add(commit.author.email)
        git_metrics["lines_added"] += commit.insertions
        git_metrics["lines_deleted"] += commit.deletions
    git_metrics["contributors"] = len(git_metrics["contributors"])
    return git_metrics

# Normalize raw metrics
def normalize_metrics(metrics, max_values):
    return {k: metrics[k] / max_values[k] if max_values[k] else 0 for k in metrics}

# Calculate complexity score
def evaluate_complexity(norm):
    return round(0.3 * norm["loc"] + 0.4 * norm["cyclomatic_complexity"] + 0.3 * norm["halstead_volume"], 4)

# Estimate effort in hours from complexity score
def estimate_effort(score, base_effort=50, scaling_factor=950):
    return round(base_effort + score * scaling_factor, 2)

# Run full evaluation for a single repo
def evaluate_repo_complexity(owner, repo_name):
    repo_path, temp_dir = clone_repo(owner, repo_name)
    language = detect_repo_language(owner, repo_name)
    print(f"Primary language: {language}")

    # Choose analysis method based on language
    if language.lower() in ["json", "yaml"]:
        print("Using JSON/YAML fallback complexity analysis...")
        code_metrics = analyze_json_yaml_complexity(repo_path)
    else:
        code_metrics = analyze_code_metrics(repo_path)

    git_metrics = analyze_git_metrics(repo_path)
    all_metrics = {**code_metrics, **git_metrics}

    max_vals = {"loc": 50000, "cyclomatic_complexity": 2000, "halstead_volume": 100000}
    normalized = normalize_metrics({
        "loc": code_metrics["loc"],
        "cyclomatic_complexity": code_metrics["cyclomatic_complexity"],
        "halstead_volume": code_metrics["halstead_volume"]
    }, max_vals)

    complexity_score = evaluate_complexity(normalized)
    effort_hours = estimate_effort(complexity_score)

    all_metrics.update({
        "owner": owner,
        "repo": repo_name,
        "language": language,
        "complexity_score": complexity_score,
        "estimated_effort_hours": effort_hours
    })

    print("\nComplexity formula: C = 0.3·(LOC_norm) + 0.4·(CC_norm) + 0.3·(HV_norm)")
    print("Effort estimation: Effort = 50 + C × 950 (in hours)")
    print(tabulate(all_metrics.items(), headers=["Metric", "Value"], tablefmt="github"))

    shutil.rmtree(temp_dir)
    return all_metrics

# Evaluate multiple repos and save output to CSV and log
def evaluate_multiple_repos(repo_list, output_csv="batch_repo_complexity.csv", log_file="batch_output.log"):
    results = []
    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, data): [f.write(data) for f in self.files]
        def flush(self): [f.flush() for f in self.files]

    with open(log_file, "w") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            for owner, repo in repo_list:
                print(f"\nEvaluating {owner}/{repo}...")
                try:
                    result = evaluate_repo_complexity(owner, repo)
                    results.append(result)
                except Exception as e:
                    print(f"Error: {e}")

            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\nSaved results to {output_csv}")

            # Plot dashboard
            plt.figure(figsize=(10, 6))
            df_sorted = df.sort_values(by="complexity_score", ascending=False)
            plt.barh(df_sorted["repo"], df_sorted["complexity_score"])
            plt.xlabel("Complexity Score")
            plt.title("Repository Complexity Scores")
            plt.tight_layout()
            plt.savefig("complexity_dashboard.png")
            print("Dashboard saved as complexity_dashboard.png")
            plt.close()



repo_list = [('neva-dev','javarel-osgi-plugin'), ('AndyObtiva','DatingNetwork'), ('lovell','sharp')]
evaluate_multiple_repos(repo_list)
