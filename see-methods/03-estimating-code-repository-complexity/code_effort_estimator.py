
import os
import math
import statistics
import ast
from typing import List, Dict

class CodeMetrics:
    """
    Collects code metrics such as LOC, Cyclomatic Complexity, Halstead Volume, Cognitive Complexity, etc.
    from a Python codebase folder.
    """

    def __init__(self, path: str):
        self.path = path
        self.loc = 0
        self.cc = []
        self.halstead_volume = []
        self.cognitive_complexity = []
        self.function_count = 0
        self.ast_depths = []

    def analyze(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".py"):
                    self._analyze_file(os.path.join(root, file))

    def _analyze_file(self, filepath: str):
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
            print(f"Error analyzing {filepath}: {e}")

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
    """
    Uses the PERT-based model wrapped around composite complexity score.
    """

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
        return round(LOC_component + CC_component + HV_component + CoCo_component + NF_component + ASTd_component, 3)

    def calculate_effort(self, C_comp: float) -> Dict[str, float]:
        a, b, c = 0.75, 1.0, 1.25
        O = a * C_comp ** 0.85
        M = b * C_comp
        P = c * C_comp ** 1.15
        effort = (O + 4 * M + P) / 6
        return {
            "C_comp": round(C_comp, 3),
            "Optimistic": round(O, 2),
            "Most Likely": round(M, 2),
            "Pessimistic": round(P, 2),
            "Effort (days)": round(effort, 2)
        }

def run_estimation_on_folder(folder_path: str):
    print(f"Analyzing: {folder_path}")
    metrics = CodeMetrics(folder_path)
    metrics.analyze()
    estimator = EffortEstimator(metrics)
    C_comp = estimator.calculate_composite_complexity()
    result = estimator.calculate_effort(C_comp)
    print("===== Results =====")
    for k, v in result.items():
        print(f"{k}: {v}")
    return result
