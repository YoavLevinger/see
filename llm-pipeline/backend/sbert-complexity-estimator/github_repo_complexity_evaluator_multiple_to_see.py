# GitHub Code Complexity Evaluator with Hybrid Static Analysis Formula
# This script evaluates GitHub repos using static code metrics (LOC, complexity),
# and applies a scientifically grounded hybrid effort estimation formula.
# It includes multiple complexity adjustment strategies: power, linear, and lookup-based.

import os
import subprocess
import shutil
import tempfile
import pandas as pd
import lizard
from pydriller import Repository
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
from contextlib import redirect_stdout
import io
import requests
import json
import yaml
import argparse



# Clone a GitHub repository to a temp directory
def clone_repo(owner, repo_name):
    temp_dir = tempfile.mkdtemp()
    repo_url = f"https://github.com/{owner}/{repo_name}.git"
    repo_path = os.path.join(temp_dir, repo_name)
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    return repo_path, temp_dir

# Detect the main language via GitHub API
def detect_repo_language(owner, repo_name):
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json().get("language", "Unknown")
    except Exception:
        pass
    return "Unknown"

# For JSON/YAML files, approximate nesting as a proxy for complexity
# Heuristic approach when source code structure is unavailable

def analyze_json_yaml_complexity(repo_path):
    loc = 0
    max_nesting = 0

    def get_max_depth(data, current_depth=1):
        if isinstance(data, dict):
            return max((get_max_depth(v, current_depth + 1) for v in data.values()), default=current_depth)
        elif isinstance(data, list):
            return max((get_max_depth(v, current_depth + 1) for v in data), default=current_depth)
        return current_depth

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.json', '.yaml', '.yml')):
                try:
                    with open(os.path.join(root, file), 'r') as f:
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

# Analyze standard source code using Lizard static analyzer

def analyze_code_metrics(repo_path):
    code_metrics = {"loc": 0, "cyclomatic_complexity": 0, "halstead_volume": 0, "function_count": 0}
    analysis = lizard.analyze([repo_path])
    for file in analysis:
        if not file.function_list:
            continue
        code_metrics["loc"] += file.nloc
        for func in file.function_list:
            code_metrics["cyclomatic_complexity"] += func.cyclomatic_complexity
            code_metrics["function_count"] += 1
            code_metrics["halstead_volume"] += func.token_count * 1.5
    code_metrics["avg_cyclomatic_complexity"] = (
        code_metrics["cyclomatic_complexity"] / code_metrics["function_count"]
        if code_metrics["function_count"] else 0
    )
    return code_metrics

# Analyze Git history metadata using PyDriller

def analyze_git_metrics(repo_path):
    git_metrics = {"commits": 0, "contributors": set(), "lines_added": 0, "lines_deleted": 0}
    for commit in Repository(repo_path).traverse_commits():
        git_metrics["commits"] += 1
        git_metrics["contributors"].add(commit.author.email)
        git_metrics["lines_added"] += commit.insertions
        git_metrics["lines_deleted"] += commit.deletions
    git_metrics["contributors"] = len(git_metrics["contributors"])
    return git_metrics

# COCOMO-based base effort estimation: Effort (PM) = a * (KLOC ^ b)
# We use nominal COCOMO II values: a=2.94, b=1.10

def base_effort_cocomo(kloc):
    return 2.94 * (kloc ** 1.10)

# Toggle between 3 complexity factor adjustment models
# 1. Power-based (default): (AvgCC / 10)^0.3 — inspired by COCOMO effort multipliers
# 2. Linear adjustment: 1 + 0.05 * (AvgCC - 10) — 5% effort increase per point above 10
# 3. Lookup table: COCOMO-style complexity driver (Very Low to Extra High)

def compute_complexity_factor(avg_cc, method="power"):
    if method == "power":
        return (avg_cc / 10) ** 0.3 if avg_cc > 0 else 1.0
    elif method == "linear":
        return 1 + 0.05 * (avg_cc - 10)
    elif method == "lookup":
        # COCOMO complexity rating table (subjective scale)
        if avg_cc < 5:
            return 0.75  # Very Low
        elif avg_cc < 10:
            return 0.88  # Low
        elif avg_cc < 15:
            return 1.00  # Nominal
        elif avg_cc < 20:
            return 1.15  # High
        else:
            return 1.30  # Very High
    return 1.0

# Final effort estimation formula
# Combines base COCOMO effort with a selected complexity factor

def estimate_effort_advanced(loc, avg_cc, method="power"):
    kloc = loc / 1000.0
    base_effort_pm = base_effort_cocomo(kloc)
    complexity_factor = compute_complexity_factor(avg_cc, method)
    adjusted_pm = base_effort_pm * complexity_factor
    return {
        "effort_person_months": round(adjusted_pm, 2),
        "effort_hours": round(adjusted_pm * 152, 2),  # assuming 152 hours per person-month
        "complexity_factor": round(complexity_factor, 3)
    }

# Full evaluation of one repository
# Includes source code analysis, effort estimation, and result printout

def evaluate_repo_complexity(owner, repo_name, complexity_mode="power"):
    repo_path, temp_dir = clone_repo(owner, repo_name)
    language = detect_repo_language(owner, repo_name)
    print(f"Primary language: {language}")

    code_metrics = analyze_json_yaml_complexity(repo_path) if language.lower() in ["json", "yaml"] \
        else analyze_code_metrics(repo_path)

    git_metrics = analyze_git_metrics(repo_path)
    all_metrics = {**code_metrics, **git_metrics}

    effort_result = estimate_effort_advanced(
        code_metrics["loc"],
        code_metrics.get("avg_cyclomatic_complexity", 0),
        method=complexity_mode
    )

    all_metrics.update({
        "owner": owner,
        "repo": repo_name,
        "language": language,
        "complexity_mode": complexity_mode,
        "complexity_score": effort_result["complexity_factor"],
        "estimated_effort_hours": effort_result["effort_hours"],
        "estimated_effort_person_months": effort_result["effort_person_months"]
    })

    print("\nHybrid Estimation Formula: Effort = 2.94 * (KLOC ^ 1.10) * ComplexityFactor")
    print(f"ComplexityFactor method: {complexity_mode}\n")
    print(tabulate(all_metrics.items(), headers=["Metric", "Value"], tablefmt="github"))

    shutil.rmtree(temp_dir)
    return all_metrics

# Batch evaluation of multiple repositories
# Outputs results to terminal, CSV, and image dashboard

def evaluate_multiple_repos(repo_list, output_csv="batch_repo_complexity.csv", log_file="batch_output.log", complexity_mode="power"):
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
                    result = evaluate_repo_complexity(owner, repo, complexity_mode=complexity_mode)
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
# repos = [("psf", "requests"), ("pallets", "flask")]
# evaluate_multiple_repos(repos, complexity_mode="lookup")

# repo_list = [('neva-dev','javarel-osgi-plugin'), ('AndyObtiva','DatingNetwork'), ('lovell','sharp')]
# evaluate_multiple_repos(repo_list)


# Unified evaluator: supports both GitHub and local path
def evaluate_codebase(source, complexity_mode="power", is_local=False):
    if is_local:
        # Analyze local directory (no clone)
        repo_path = source
        language = "Unknown"
        print(f"Analyzing local codebase: {repo_path}")
        code_metrics = analyze_json_yaml_complexity(repo_path) if repo_path.endswith(('.json', '.yaml', '.yml')) \
            else analyze_code_metrics(repo_path)

        # Only analyze Git metrics if it's a valid Git repo
        if os.path.isdir(os.path.join(repo_path, ".git")):
            git_metrics = analyze_git_metrics(repo_path)
        else:
            git_metrics = {
                "commits": 0,
                "contributors": 0,
                "lines_added": 0,
                "lines_deleted": 0
            }

        all_metrics = {**code_metrics, **git_metrics}
        effort_result = estimate_effort_advanced(
            code_metrics["loc"],
            code_metrics.get("avg_cyclomatic_complexity", 0),
            method=complexity_mode
        )
        all_metrics.update({
            "owner": "local",
            "repo": os.path.basename(repo_path),
            "language": language,
            "complexity_mode": complexity_mode,
            "complexity_score": effort_result["complexity_factor"],
            "estimated_effort_hours": effort_result["effort_hours"],
            "estimated_effort_person_months": effort_result["effort_person_months"]
        })

        print("\nHybrid Estimation Formula: Effort = 2.94 * (KLOC ^ 1.10) * ComplexityFactor")
        print(f"ComplexityFactor method: {complexity_mode}\n")
        print(tabulate(all_metrics.items(), headers=["Metric", "Value"], tablefmt="github"))
        return all_metrics
    else:
        owner, repo = source
        return evaluate_repo_complexity(owner, repo, complexity_mode)

# CLI handler
# if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Estimate software development effort from code.")
    # subparsers = parser.add_subparsers(dest="mode", required=True)
    #
    # # GitHub mode
    # parser_github = subparsers.add_parser("github", help="Analyze GitHub repository")
    # parser_github.add_argument("owner", type=str, help="GitHub owner name")
    # parser_github.add_argument("repo", type=str, help="GitHub repository name")
    # parser_github.add_argument("--complexity", choices=["power", "linear", "lookup"], default="power")
    #
    # # Local mode
    # parser_local = subparsers.add_parser("local", help="Analyze local code folder")
    # parser_local.add_argument("path", type=str, help="Path to local source code directory")
    # parser_local.add_argument("--complexity", choices=["power", "linear", "lookup"], default="power")
    #
    # args = parser.parse_args()
    #
    # if args.mode == "github":
    #     evaluate_codebase((args.owner, args.repo), complexity_mode=args.complexity, is_local=False)
    # elif args.mode == "local":
    #     evaluate_codebase(args.path, complexity_mode=args.complexity, is_local=True)


# Example usage:
# repos = [("psf", "requests"), ("pallets", "flask")]
# evaluate_multiple_repos(repos, complexity_mode="lookup")

# repo_list = [('neva-dev','javarel-osgi-plugin'), ('AndyObtiva','DatingNetwork'), ('lovell','sharp')]
# evaluate_multiple_repos(repo_list , complexity_mode="lookup")

# evaluate_codebase(('AndyObtiva','DatingNetwork'), complexity_mode="linear", is_local=False)
#
#
# code_path1 = "/home/yoav-levinger/projects/source/2nd/see/llm-pipeline/generated-code/a6466ac3-105e-419b-9e9f-c1d35ba546c7"
# evaluate_codebase(code_path1, complexity_mode="linear", is_local=True)