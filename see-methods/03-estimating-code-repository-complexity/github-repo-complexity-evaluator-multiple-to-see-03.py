# GitHub Code Complexity Evaluator with Hybrid Static Analysis Formula
# This script clones GitHub repos, analyzes static code metrics (LOC, complexity),
# applies a hybrid scientific effort estimation model based on COCOMO and complexity,
# and outputs detailed effort estimates.

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

# Clone a GitHub repository into a temporary directory
def clone_repo(owner, repo_name):
    temp_dir = tempfile.mkdtemp()
    repo_url = f"https://github.com/{owner}/{repo_name}.git"
    repo_path = os.path.join(temp_dir, repo_name)
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    return repo_path, temp_dir

# Detect the main language from GitHub API
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

# Analyze JSON or YAML files to estimate nesting and LOC
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
        "function_count": 0,
        "avg_cyclomatic_complexity": 0
    }

# Analyze source code using Lizard
def analyze_code_metrics(repo_path):
    code_metrics = {
        "loc": 0,
        "cyclomatic_complexity": 0,
        "halstead_volume": 0,
        "function_count": 0
    }
    analysis = lizard.analyze([repo_path])
    for file in analysis:
        if not file.function_list:
            continue
        code_metrics["loc"] += file.nloc
        for function in file.function_list:
            code_metrics["cyclomatic_complexity"] += function.cyclomatic_complexity
            code_metrics["function_count"] += 1
            code_metrics["halstead_volume"] += function.token_count * 1.5
    code_metrics["avg_cyclomatic_complexity"] = (
        code_metrics["cyclomatic_complexity"] / code_metrics["function_count"]
        if code_metrics["function_count"] else 0
    )
    return code_metrics

# Estimate complexity adjustment factor (default baseline complexity = 10)
def compute_complexity_factor(avg_cc, baseline=10.0):
    return (avg_cc / baseline) ** 0.3 if avg_cc > 0 else 1.0

# Hybrid effort estimation formula: COCOMO II + complexity factor
# Effort (PM) = 2.94 * (KLOC^1.10) * ComplexityFactor
# 1 PM ~ 152 hours (standard)
def estimate_effort_advanced(loc, avg_cc):
    kloc = loc / 1000.0
    base_effort_pm = 2.94 * (kloc ** 1.10)
    complexity_factor = compute_complexity_factor(avg_cc)
    adjusted_effort_pm = base_effort_pm * complexity_factor
    return {
        "effort_person_months": round(adjusted_effort_pm, 2),
        "effort_hours": round(adjusted_effort_pm * 152, 2),
        "complexity_factor": round(complexity_factor, 3)
    }

# Main repository evaluation workflow
def evaluate_repo_complexity(owner, repo_name):
    repo_path, temp_dir = clone_repo(owner, repo_name)
    language = detect_repo_language(owner, repo_name)
    print(f"Primary language: {language}")

    if language.lower() in ["json", "yaml"]:
        print("Using JSON/YAML fallback complexity analysis...")
        code_metrics = analyze_json_yaml_complexity(repo_path)
    else:
        code_metrics = analyze_code_metrics(repo_path)

    git_metrics = analyze_git_metrics(repo_path)
    all_metrics = {**code_metrics, **git_metrics}

    effort_result = estimate_effort_advanced(code_metrics["loc"], code_metrics.get("avg_cyclomatic_complexity", 0))

    all_metrics.update({
        "owner": owner,
        "repo": repo_name,
        "language": language,
        "complexity_score": effort_result["complexity_factor"],
        "estimated_effort_hours": effort_result["effort_hours"],
        "estimated_effort_person_months": effort_result["effort_person_months"]
    })

    print("\nHybrid Estimation Formula: Effort = 2.94 * (KLOC ^ 1.10) * ComplexityFactor")
    print("ComplexityFactor = (AvgCyclomatic / 10)^0.3\n")
    print(tabulate(all_metrics.items(), headers=["Metric", "Value"], tablefmt="github"))

    shutil.rmtree(temp_dir)
    return all_metrics

# Analyze Git commit data using PyDriller
def analyze_git_metrics(repo_path):
    git_metrics = {"commits": 0, "contributors": set(), "lines_added": 0, "lines_deleted": 0}
    for commit in Repository(repo_path).traverse_commits():
        git_metrics["commits"] += 1
        git_metrics["contributors"].add(commit.author.email)
        git_metrics["lines_added"] += commit.insertions
        git_metrics["lines_deleted"] += commit.deletions
    git_metrics["contributors"] = len(git_metrics["contributors"])
    return git_metrics

# Batch evaluation across multiple repos with output to terminal and log file
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

            plt.figure(figsize=(10, 6))
            df_sorted = df.sort_values(by="estimated_effort_hours", ascending=False)
            plt.barh(df_sorted["repo"], df_sorted["estimated_effort_hours"])
            plt.xlabel("Estimated Effort (hours)")
            plt.title("Repository Effort Estimations")
            plt.tight_layout()
            plt.savefig("effort_dashboard.png")
            print("Dashboard saved as effort_dashboard.png")
            plt.close()

# Example usage:
# repos = [("psf", "requests"), ("pallets", "flask"), ("neva-dev", "javarel-osgi-plugin")]
# evaluate_multiple_repos(repos)
