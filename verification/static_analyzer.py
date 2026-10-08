"""Static Analysis for migrated GDScript files.

Computes:
- Syntax Error Rate (%) via Godot 4 headless --check-only
- AST Edit Distance between generated and ground-truth scripts
- CodeBLEU Score (Supports native GDScript module or standard Python mapping)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import CODEBLEU_LANG, CODEBLEU_WEIGHTS, USE_NATIVE_GDSCRIPT_CODEBLEU
from verification.gdscript_codebleu import calc_gdscript_codebleu

logger = logging.getLogger(__name__)


class StaticAnalyzer:
    """Performs static analysis and code-quality metrics on migrated scripts."""

    # ---------------------------------------------------------------- #
    #  Syntax Checking (delegates to Docker harness)
    # ---------------------------------------------------------------- #

    @staticmethod
    def check_syntax(
        docker_harness,
        project_dir: str,
        script_path: str,
    ) -> Dict[str, Any]:
        """Run ``godot4 --headless --check-only`` inside the container."""
        cmd = f"godot4 --headless --check-only --script {script_path}"
        process_res = docker_harness.run_in_container(
            project_dir=project_dir,
            command=cmd,
        )
        passed = process_res.returncode == 0
        return {
            "passed": passed,
            "returncode": process_res.returncode,
            "errors": process_res.stderr if not passed else "",
        }

    # ---------------------------------------------------------------- #
    #  AST Edit Distance
    # ---------------------------------------------------------------- #

    @staticmethod
    def compute_ast_edit_distance(
        generated_code: str,
        reference_code: str,
    ) -> int:
        """Compute tree edit distance between two GDScript sources."""
        try:
            return StaticAnalyzer._tree_sitter_edit_distance(
                generated_code, reference_code
            )
        except Exception as exc:
            logger.debug(
                "tree-sitter TED unavailable (%s); using line-level proxy.",
                exc,
            )
            return StaticAnalyzer._line_edit_distance(
                generated_code, reference_code
            )

    @staticmethod
    def _tree_sitter_edit_distance(code_a: str, code_b: str) -> int:
        """TED via tree-sitter GDScript parser + apted."""
        import tree_sitter_language_pack as tslp

        # Uses native GDScript parser when available, falls back to Python mapping
        lang_parser = "gdscript" if USE_NATIVE_GDSCRIPT_CODEBLEU else "python"
        parser = tslp.get_parser(lang_parser)
        tree_a = parser.parse(code_a.encode())
        tree_b = parser.parse(code_b.encode())

        def to_bracket(node) -> str:
            label = node.type.replace("{", "_").replace("}", "_")
            children = "".join(to_bracket(c) for c in node.children)
            return "{" + label + children + "}"

        try:
            from apted import APTED
            from apted.helpers import Tree

            a_str = to_bracket(tree_a.root_node)
            b_str = to_bracket(tree_b.root_node)
            t1 = Tree.from_text(a_str)
            t2 = Tree.from_text(b_str)
            return APTED(t1, t2).compute_edit_distance()
        except ImportError:
            logger.debug("apted not installed; falling back to line diff.")
            return StaticAnalyzer._line_edit_distance(code_a, code_b)

    @staticmethod
    def _line_edit_distance(code_a: str, code_b: str) -> int:
        """Levenshtein distance on lines as a rough proxy."""
        lines_a = code_a.splitlines()
        lines_b = code_b.splitlines()
        n, m = len(lines_a), len(lines_b)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, m + 1):
                cost = 0 if lines_a[i - 1] == lines_b[j - 1] else 1
                dp[j], prev = (
                    min(dp[j] + 1, dp[j - 1] + 1, prev + cost),
                    dp[j],
                )
        return dp[m]

    # ---------------------------------------------------------------- #
    #  CodeBLEU Selection (Native GDScript vs Python mapping)
    # ---------------------------------------------------------------- #

    @staticmethod
    def compute_codebleu(
        generated_code: str,
        reference_code: str,
        use_native_gdscript: bool = USE_NATIVE_GDSCRIPT_CODEBLEU,
        lang: str = CODEBLEU_LANG,
        weights: Tuple[float, ...] = CODEBLEU_WEIGHTS,
    ) -> Dict[str, float]:
        """Calculate CodeBLEU score using either native GDScript AST parser or standard Python mapping."""
        try:
            if use_native_gdscript:
                return calc_gdscript_codebleu(
                    predictions=[generated_code],
                    references=[[reference_code]],
                    weights=weights,
                )
            else:
                from codebleu import calc_codebleu

                return calc_codebleu(
                    references=[[reference_code]],
                    predictions=[generated_code],
                    lang=lang,
                    weights=weights,
                )
        except Exception as exc:
            logger.error("CodeBLEU computation failed: %s", exc)
            return {"codebleu": 0.0, "error": str(exc)}

    # ---------------------------------------------------------------- #
    #  Aggregate
    # ---------------------------------------------------------------- #

    def analyse_file(
        self,
        generated_code: str,
        reference_code: str,
        docker_harness=None,
        container_script_path: Optional[str] = None,
        project_dir: Optional[str] = None,
        use_native_gdscript: bool = USE_NATIVE_GDSCRIPT_CODEBLEU,
    ) -> Dict[str, Any]:
        """Run all static metrics on a single file."""
        metrics: Dict[str, Any] = {}

        if docker_harness and container_script_path and project_dir:
            syntax = self.check_syntax(docker_harness, project_dir, container_script_path)
            metrics["syntax_passed"] = syntax["passed"]
            metrics["syntax_errors"] = syntax["errors"]
        else:
            metrics["syntax_passed"] = None
            metrics["syntax_errors"] = "Docker harness not provided."

        metrics["ast_edit_distance"] = self.compute_ast_edit_distance(
            generated_code, reference_code
        )

        cb = self.compute_codebleu(
            generated_code, reference_code, use_native_gdscript=use_native_gdscript
        )
        metrics["codebleu"] = cb.get("codebleu", 0.0)
        metrics["codebleu_detail"] = cb

        return metrics