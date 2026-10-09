"""
Native GDScript CodeBLEU Evaluation Module.

Computes CodeBLEU components using dedicated GDScript Tree-Sitter AST parsing
and GDScript keyword weighting.
"""

import logging
from collections import Counter
from typing import Dict, List, Set, Tuple

import tree_sitter_language_pack as tslp

logger = logging.getLogger(__name__)


def _get_gdscript_keywords() -> Set[str]:
    """Dynamically extracts keywords and built-in type symbols from GDScript Tree-Sitter grammar."""
    try:
        language = tslp.get_language("gdscript")
        keywords = set()
        # Tree-sitter modern API uses node_kind instead of node_type
        for i in range(language.node_kind_count):
            is_named = language.node_kind_is_named(i)
            kind_name = language.node_kind_for_id(i)
            if kind_name and not is_named and kind_name.isalpha():
                keywords.add(kind_name)
        return keywords
    except Exception as exc:
        logger.warning("Failed to extract GDScript keywords dynamically: %s", exc)
        return set()


def _extract_ast_subtrees(node) -> List[str]:
    """Recursively serializes AST sub-trees as string representations."""
    subtrees = []

    def _traverse(n):
        if not n.children:
            return n.type
        child_reprs = [_traverse(c) for c in n.children]
        repr_str = f"({n.type} {' '.join(child_reprs)})"
        subtrees.append(repr_str)
        return repr_str

    _traverse(node)
    return subtrees


def compute_gdscript_ast_match(pred_code: str, ref_code: str) -> float:
    """Computes AST sub-tree match ratio using native GDScript Tree-Sitter parser."""
    try:
        parser = tslp.get_parser("gdscript")
        tree_pred = parser.parse(pred_code.encode())
        tree_ref = parser.parse(ref_code.encode())

        pred_subtrees = _extract_ast_subtrees(tree_pred.root_node)
        ref_subtrees = _extract_ast_subtrees(tree_ref.root_node)

        if not ref_subtrees or not pred_subtrees:
            return 0.0

        ref_counts = Counter(ref_subtrees)
        pred_counts = Counter(pred_subtrees)

        overlap = sum(min(count, ref_counts[st]) for st, count in pred_counts.items())
        return overlap / float(len(pred_subtrees))
    except Exception as exc:
        logger.warning("Native GDScript AST match failed: %s", exc)
        return 0.0


def compute_ngram_match(pred_tokens: List[str], ref_tokens: List[str], n: int = 4) -> float:
    """Computes standard BLEU n-gram precision score."""
    scores = []
    for i in range(1, n + 1):
        pred_ngrams = [tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens)-i+1)]
        ref_ngrams = [tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1)]

        if not pred_ngrams or not ref_ngrams:
            scores.append(0.0)
            continue

        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)

        overlap = sum(min(count, ref_counts[ng]) for ng, count in pred_counts.items())
        scores.append(overlap / float(len(pred_ngrams)))

    return sum(scores) / len(scores) if scores else 0.0


def compute_weighted_ngram_match(
    pred_tokens: List[str],
    ref_tokens: List[str],
    keywords: Set[str] = None,
    n: int = 4,
) -> float:
    """Computes keyword-weighted n-gram match score using GDScript keywords."""
    if keywords is None:
        keywords = _get_gdscript_keywords()

    scores = []
    for i in range(1, n + 1):
        pred_ngrams = [tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens)-i+1)]
        ref_ngrams = [tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1)]

        if not pred_ngrams or not ref_ngrams:
            scores.append(0.0)
            continue

        ref_counts = Counter(ref_ngrams)
        pred_counts = Counter(pred_ngrams)

        weighted_overlap = 0.0
        total_weight = 0.0

        for ng, count in pred_counts.items():
            weight = 5.0 if any(tok in keywords for tok in ng) else 1.0
            matched = min(count, ref_counts[ng])
            weighted_overlap += matched * weight
            total_weight += count * weight

        scores.append(weighted_overlap / total_weight if total_weight > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def calc_gdscript_codebleu(
    predictions: List[str],
    references: List[List[str]],
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> Dict[str, float]:
    """Calculates native GDScript CodeBLEU metric.

    Combines BLEU, keyword-weighted BLEU, and Tree-Sitter GDScript AST match.
    """
    if not predictions or not references:
        return {"codebleu": 0.0}

    pred_code = predictions[0]
    ref_code = references[0][0]

    pred_tokens = pred_code.split()
    ref_tokens = ref_code.split()

    keywords = _get_gdscript_keywords()

    ngram_score = compute_ngram_match(pred_tokens, ref_tokens)
    weighted_ngram_score = compute_weighted_ngram_match(pred_tokens, ref_tokens, keywords=keywords)
    ast_match_score = compute_gdscript_ast_match(pred_code, ref_code)

    # Re-normalize weights across the 3 calculated sub-components
    w1, w2, w3 = weights[:3]
    w_sum = w1 + w2 + w3

    codebleu_score = (
        (w1 * ngram_score) +
        (w2 * weighted_ngram_score) +
        (w3 * ast_match_score)
    ) / w_sum

    return {
        "codebleu": round(codebleu_score, 4),
        "ngram_match_score": round(ngram_score, 4),
        "weighted_ngram_match_score": round(weighted_ngram_score, 4),
        "syntax_match_score": round(ast_match_score, 4),
    }