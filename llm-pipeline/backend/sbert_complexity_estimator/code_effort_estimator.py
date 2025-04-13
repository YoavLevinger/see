import os
import math
import statistics
import subprocess
import json
import ast
import re
from typing import List, Dict

# --- Language-specific heuristics for error-tolerant complexity estimation ---
def heuristic_js_ts(code: str) -> dict:
    return {
        "functions": len(re.findall(r'function\s+\w+|\w+\s*=>', code)),
        "callbacks": len(re.findall(r'\.then\s*\(|async\s+function|await\s+', code)),
        "dynamic_features": len(re.findall(r'\beval\b|\bFunction\s*\(', code)),
        "dom_events": len(re.findall(r'document\.querySelector|addEventListener', code))
    }

def heuristic_java(code: str) -> dict:
    return {
        "classes": len(re.findall(r'\bclass\s+\w+', code)),
        "annotations": len(re.findall(r'@\w+', code)),
        "threads": len(re.findall(r'synchronized|Thread|Runnable', code)),
        "generics": len(re.findall(r'<[^<>]+<[^<>]+>>', code))
    }

def heuristic_cpp(code: str) -> dict:
    return {
        "pointers": len(re.findall(r'\*|\&|->', code)),
        "macros": len(re.findall(r'#define|#ifdef|#ifndef', code)),
        "memory": len(re.findall(r'malloc|free|new|delete', code)),
        "includes": len(re.findall(r'#include', code))
    }

def heuristic_python(code: str) -> dict:
    return {
        "dynamic_features": len(re.findall(r'eval|exec|__getattr__', code)),
        "metaprogramming": len(re.findall(r'type\s*\(|__new__', code)),
        "nested_comprehensions": len(re.findall(r'\[.*\n*\[.*\].*\]', code, re.DOTALL))
    }

def detect_language_heuristics(code: str, language: str) -> dict:
    lang = language.lower()
    if lang in ['javascript', 'typescript']: return heuristic_js_ts(code)
    if lang == 'java': return heuristic_java(code)
    if lang in ['c', 'cpp']: return heuristic_cpp(code)
    if lang == 'python': return heuristic_python(code)
    return {}
# ------------------------------------------------------------------------------

class CodeMetrics:
    def __init__(self, path: str):
        self.path = path
        self.loc = 0
        self.cc = []
        self.halstead_volume = []
        self.cognitive_complexity = []
        self.function_count = 0
        self.ast_depths = []

    def analyze(self):
        print(f"🌍 Analyzing codebase at: {self.path}")
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                lang = self._detect_language(file)
                print(f"🔍 [{lang}] Analyzing: {file_path}")

                if lang == "python":
                    self._analyze_python_ast(file_path)
                elif lang in {"javascript", "java", "go", "cpp", "csharp"}:
                    self._analyze_lizard(file_path, lang)
                else:
                    self._analyze_heuristic(file_path, lang)

        print("📊 Summary Metrics:")
        print(f"- LOC: {self.loc}")
        print(f"- Avg CC: {statistics.mean(self.cc) if self.cc else 0}")
        print(f"- Avg Halstead: {statistics.mean(self.halstead_volume) if self.halstead_volume else 0}")
        print(f"- Avg Cognitive: {statistics.mean(self.cognitive_complexity) if self.cognitive_complexity else 0}")
        print(f"- Avg AST Depth: {statistics.mean(self.ast_depths) if self.ast_depths else 0}")
        print(f"- Functions: {self.function_count}")

    def _detect_language(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        return {
            ".py": "python", ".js": "javascript", ".java": "java",
            ".go": "go", ".c": "c", ".cpp": "cpp", ".cs": "csharp"
        }.get(ext, "unknown")

    def _analyze_python_ast(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
                self.loc += sum(1 for line in source.splitlines() if line.strip())
                tree = ast.parse(source)
                self.function_count += len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
                self.ast_depths.append(self._calculate_ast_depth(tree))
                self.cc.append(self._calculate_cyclomatic_complexity(tree))
                self.halstead_volume.append(self._approximate_halstead_volume(source))
                self.cognitive_complexity.append(self._approximate_cognitive_complexity(tree))
        except Exception as e:
            print(f"⚠️ AST parse failed on {filepath}: {e}")
            self._fallback_heuristic(filepath, "python")

    def _analyze_lizard(self, filepath, lang):
        try:
            result = subprocess.run(
                ["lizard", "-l", lang, "-C", "10", "-j", filepath],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
            )
            data = json.loads(result.stdout)
            for file in data.get("files", []):
                self.loc += file.get("nloc", 0)
                for func in file.get("functions", []):
                    self.cc.append(func.get("cyclomatic_complexity", 1))
                    self.function_count += 1
        except Exception as e:
            print(f"⚠️ Lizard failed on {filepath}: {e}")
            self._fallback_heuristic(filepath, lang)

    def _analyze_heuristic(self, filepath, lang):
        self._fallback_heuristic(filepath, lang)

    def _fallback_heuristic(self, filepath, lang):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
                self.loc += sum(1 for line in source.splitlines() if line.strip())
                h = detect_language_heuristics(source, lang)
                score = sum(h.values())
                self.cc.append(min(1 + score, 10))
                self.function_count += h.get("functions", h.get("classes", 1))
                self.halstead_volume.append(score * 2)
                self.cognitive_complexity.append(score)
        except Exception as e:
            print(f"⚠️ Heuristic parse failed on {filepath}: {e}")

    def _calculate_ast_depth(self, node: ast.AST, level: int = 0) -> int:
        if not list(ast.iter_child_nodes(node)):
            return level
        return max(self._calculate_ast_depth(child, level + 1) for child in ast.iter_child_nodes(node))

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        return 1 + sum(1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.And, ast.Or)))

    def _approximate_halstead_volume(self, source: str) -> float:
        tokens = source.split()
        ops = [tok for tok in tokens if tok in ['+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '>=', '<=']]
        opr = [tok for tok in tokens if tok not in ops]
        n1, n2 = len(set(ops)), len(set(opr))
        N1, N2 = len(ops), len(opr)
        try:
            return (N1 + N2) * math.log2(n1 + n2) if (n1 + n2) > 0 else 0
        except:
            return 0

    def _approximate_cognitive_complexity(self, tree: ast.AST) -> int:
        def _score(node, depth=1):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.FunctionDef)):
                return depth + sum(_score(child, depth + 1) for child in ast.iter_child_nodes(node))
            return sum(_score(child, depth) for child in ast.iter_child_nodes(node))
        return _score(tree)


class EffortEstimator:
    def __init__(self, metrics: CodeMetrics):
        self.metrics = metrics

    def calculate_composite_complexity(self) -> float:
        w1, w2, w3, w4, w5, w6 = 1, 2, 1, 1.5, 1, 1
        LOC_component = w1 * (self.metrics.loc / 1000)
        CC_component = w2 * statistics.mean(self.metrics.cc) if self.metrics.cc else 0
        HV_component = w3 * statistics.mean(self.metrics.halstead_volume) if self.metrics.halstead_volume else 0
        CoCo_component = w4 * statistics.mean(self.metrics.cognitive_complexity) if self.metrics.cognitive_complexity else 0
        NF_component = w5 * math.log(self.metrics.function_count + 1)
        ASTd_component = w6 * statistics.mean(self.metrics.ast_depths) if self.metrics.ast_depths else 0

        C_comp = round(LOC_component + CC_component + HV_component + CoCo_component + NF_component + ASTd_component, 3)
        print(f"🧮 Composite Complexity (C_comp): {C_comp}")
        return C_comp

    def calculate_effort(self, C_comp: float) -> Dict[str, float]:
        a, b, c = 0.75, 1.0, 1.25
        O = a * C_comp ** 0.85
        M = b * C_comp
        P = c * C_comp ** 1.15
        effort = (O + 4 * M + P) / 6

        print("📈 PERT-Based Effort Estimation:")
        print(f"  Optimistic: {O:.2f} | Most Likely: {M:.2f} | Pessimistic: {P:.2f} | Effort (days): {effort:.2f}")

        return {
            "C_comp": round(C_comp, 3),
            "Optimistic": round(O, 2),
            "Most Likely": round(M, 2),
            "Pessimistic": round(P, 2),
            "Effort (days)": round(effort, 2)
        }


def run_estimation_on_folder(folder_path: str):
    print(f"📂 Running academic estimation on: {folder_path}")
    metrics = CodeMetrics(folder_path)
    metrics.analyze()
    estimator = EffortEstimator(metrics)
    C_comp = estimator.calculate_composite_complexity()
    result = estimator.calculate_effort(C_comp)

    print("===== Final Result Summary =====")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result
