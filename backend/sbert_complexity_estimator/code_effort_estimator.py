import os
import math
import statistics
import subprocess
import json
import ast
import re
import xml.etree.ElementTree as ET
import subprocess
import tempfile
from typing import Dict, Any, List
import lizard

#verbosity/density of different languages (e.g., Java code tends to be more verbose than Python,
# so Java gets a slightly reduced multiplier)

LANGUAGE_WEIGHTS = {
    'python': 1.0,
    'java': 0.9,
    'javascript': 0.95,
    'csharp': 0.9,
    'cpp': 1.1,
    'c': 1.1,
    'go': 0.95,
    'typescript': 0.95,
    'ruby': 0.95,
    'php': 0.9,
    'kotlin': 0.9,
    'scala': 0.9,
    'swift': 0.9,
    'objective-c': 1.1,
    'r': 0.95,
    'lua': 0.9,
    'groovy': 0.9,
    'dart': 0.9,
    'rust': 1.0,
    'unknown': 1.0,
}

EXTENSION_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.java': 'java',
    '.cs': 'csharp',
    '.cpp': 'cpp',
    '.cxx': 'cpp',
    '.cc': 'cpp',
    '.c': 'c',
    '.go': 'go',
    '.rb': 'ruby',
    '.php': 'php',
    '.rs': 'rust',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.swift': 'swift',
    '.m': 'objective-c',
    '.mm': 'objective-c',
    '.scala': 'scala',
    '.lua': 'lua',
    '.groovy': 'groovy',
    '.dart': 'dart',
    '.r': 'r'
}

def detect_language_by_extension(filename):
    ext = os.path.splitext(filename)[1].lower()
    return EXTENSION_MAP.get(ext, 'unknown')




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
        CODE_FILE_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rb'}
        NOT_CODE_FILE_EXTENSIONS = {
            # Documentation & text
            '.md', '.rst', '.txt', '.doc', '.docx', '.pdf',

            # Config & metadata
            '.json', '.yaml', '.yml', '.ini', '.cfg', '.toml', '.lock', '.xml', '.html', '.css',

            # Binary / Media / Fonts
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico',
            '.mp4', '.mp3', '.wav', '.ogg', '.mov',
            '.woff', '.woff2', '.ttf', '.eot',

            # Archives & packages
            '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz',

            # Logs & database
            '.log', '.sql', '.sqlite', '.db',

            # Compiled & bytecode
            '.pyc', '.pyo', '.class', '.exe', '.dll', '.so', '.o', '.a', '.dylib',

            # Virtual environments / installers
            '.whl', '.egg', '.deb', '.rpm', '.msi',

            # Miscellaneous
            '.csv', '.tsv', '.xlsx', '.xls', '.ppt', '.pptx', '.sample','.sln',

            #additinal:
            '.csproj', '.aidl', '.json', '.min.js','.gitignore','.out', '.properties','.pem','LICENSE', 'README', '.gradle',
            '.project','.gitattributes','.classpath','.babelrc','.opts','gradlew','.bat','.props','.targets','.cache',
        }
        IGNORED_FOLDERS = {'.git', 'docs', 'Resources', 'static', '__pycache__'}
        print(f"Analyzing codebase at: {self.path}")
        for root, dirs, files in os.walk(self.path):

            # Skip .git and other hidden/system folders
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            dirs[:] = [d for d in dirs if d not in IGNORED_FOLDERS] #TODO
            for file in files:
                if any(file.endswith(ext) for ext in NOT_CODE_FILE_EXTENSIONS):
                    continue  # Skip non-code files
                # Skip files inside .git or any hidden/system directory
                file_path = os.path.join(root, file)
                # if '/.git/' in file_path or file_path.endswith('.git'):
                #     continue
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
            analysis = lizard.analyze_file(filepath)
            if not analysis.function_list:
                print(f"⚠️ No functions parsed by Lizard in: {filepath}")
                return

            for function in analysis.function_list:
                self.loc += function.length
                self.cc.append(function.cyclomatic_complexity)
                self.halstead_volume.append(function.token_count)  # approximation
                self.cognitive_complexity.append(function.cognitive_complexity if hasattr(function,
                                                                                          "cognitive_complexity") else function.cyclomatic_complexity)
                self.function_count += 1

        except Exception as e:
            print(f"⚠️ Native Lizard failed on {filepath}: {e}")
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
        loc_score = self.normalize_loc(self.metrics.loc)
        cc_score = self.normalize_cyclomatic(statistics.mean(self.metrics.cc)) if self.metrics.cc else 0
        halstead_score = self.normalize_halstead(
            statistics.mean(self.metrics.halstead_volume)) if self.metrics.halstead_volume else 0
        cognitive_score = self.normalize_cognitive(
            statistics.mean(self.metrics.cognitive_complexity)) if self.metrics.cognitive_complexity else 0
        func_score = self.normalize_functions(self.metrics.function_count)
        ast_depth_score = self.normalize_ast_depth(statistics.mean(self.metrics.ast_depths)) if self.metrics.ast_depths else 0

        weights = { #w1, w2, w3, w4, w5, w6
            # w1, w2, w3, w4, w5, w6 = 1.2, 1.6, 1.2, 1.2, 1.0, 1.0
            "LOC": 6.5,
            "Functions": 2.0,
            "Avg Cyclomatic Complexity": 1.2,
            "Avg Halstead Volume": 1.2,
            "Avg Cognitive Complexity": 1.0,
            "Avg AST Depth": 1.0
        }

        composite_score = ( #C_comp
                loc_score * weights["LOC"] +
                func_score * weights["Functions"] +
                cc_score * weights["Avg Cyclomatic Complexity"] +
                halstead_score * weights["Avg Halstead Volume"] +
                cognitive_score * weights["Avg Cognitive Complexity"] +
                ast_depth_score * weights["Avg AST Depth"]
        )


        # Detect language from first file (or default to 'python')
        lang = 'python'
        for root, dirs, files in os.walk(self.metrics.path):
            for f in files:
                lang = detect_language_by_extension(f)
                if lang:
                    break
            break

        lang_multiplier = LANGUAGE_WEIGHTS.get(lang.lower(), 1.0)

        # We had gone through ~200 code repositories evaluation and concluded this for now
        empirical_evaluator_factor = 2.9


        return round(composite_score * empirical_evaluator_factor * lang_multiplier, 3)



    def normalize_loc(self, loc: float) -> float:
        if loc < 2000:
            return loc / 4000  # was /2000
        elif loc < 5000:
            return 0.5 + (loc - 2000) / 9000  # was 6000
        elif loc < 15000:
            return 0.8 + (loc - 5000) / 25000  # was 35000
        return 1.0

    def normalize_functions(self, count: float) -> float:
        if count < 20:
            return count / 40
        elif count < 100:
            return 0.5 + (count - 20) / 160
        elif count < 300:
            return 0.8 + (count - 100) / 1000
        return 1.0

    def normalize_halstead(self, volume: float) -> float:
        if volume < 200:
            return volume / 400
        elif volume < 1000:
            return 0.5 + (volume - 200) / 1600
        elif volume < 3000:
            return 0.8 + (volume - 1000) / 8000
        return 1.0

    def normalize_cognitive(self, cognitive: float) -> float:
        if cognitive < 5:
            return cognitive / 10
        elif cognitive < 15:
            return 0.5 + (cognitive - 5) / 20
        elif cognitive < 50:
            return 0.8 + (cognitive - 15) / 140
        return 1.0

    # def normalize_ast_depth(self, depth: float) -> float:
    #     return min(depth / 15, 1.0)

    def normalize_ast_depth(self, depth: float) -> float:
        if depth < 8:
            return depth / 16
        elif depth < 20:
            return 0.5 + (depth - 8) / 24
        return 1.0

    # def normalize_cyclomatic(self, cc: float) -> float:
    #     return min(cc / 10, 1.0)

    def normalize_cyclomatic(self, cc: float) -> float:
        if cc < 5:
            return cc / 10
        elif cc < 15:
            return 0.5 + (cc - 5) / 20
        return 1.0



    def calculate_effort(self, C_comp: float) -> Dict[str, float]:
        # Academic PERT estimation
        a, b, c = 0.85, 1.0, 2.0
        O = a * C_comp ** 0.95
        M = b * C_comp
        P = c * C_comp ** 1.1
        effort_days = (O + 4 * M + P) / 6

        # LOC adjustment — adds more time if the repo is large
        loc = self.metrics.loc
        # loc_adjustment = min(loc / 40.0, 200.0) * 0.02  # 40 LOC/day baseline
        loc_adjustment = (loc ** 0.5) * 0.03  # nonlinear, increases more naturally
        adjusted_effort_days = effort_days + loc_adjustment

        # Convert to hours (with moderate overhead)
        effort_hours = round(adjusted_effort_days * 8 * 1.1, 2)

        print("PERT-Based Effort Estimation:")
        print(f"  Optimistic: {O:.2f} | Most Likely: {M:.2f} | Pessimistic: {P:.2f} | Effort (days): {adjusted_effort_days:.2f}")

        return {
            "C_comp": round(C_comp, 3),
            "Optimistic": round(O, 2),
            "Most Likely": round(M, 2),
            "Pessimistic": round(P, 2),
            "Effort (days)": round(adjusted_effort_days, 2),
            "hours": effort_hours
        }

def run_estimation_on_folder(folder_path: str):
    print(f"Running estimation on: {folder_path}")
    metrics = CodeMetrics(folder_path)
    metrics.analyze()
    estimator = EffortEstimator(metrics)
    C_comp = estimator.calculate_composite_complexity()
    result = estimator.calculate_effort(C_comp)

    print("===== Final Result Summary =====")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result


if __name__ == "__main__":
    folder_to_analyze = "/home/yoav-levinger/Documents/private/2nd degree/Final Project/solution V2/llm-pipeline"
    run_estimation_on_folder(folder_to_analyze)
