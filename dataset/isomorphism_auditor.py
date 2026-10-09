"""Isomorphism Auditor for Godot project pairs.

Verifies structural equivalence between legacy (3.x) and target (4.x) demo projects:
1. Critical Path Alignment – checks .gd and .tscn presence (case-insensitive).
2. Graph-Based Isomorphism – checks tree topology invariance with FULL sibling reordering immunity
   using AHU-style canonical subtree signatures.
3. 1:1 Node Type/Instance Shifts – flags structural matches with modified types/instances or
   renamed nodes for manual review.
4. Non-Critical Path Warnings – tracks extra/missing assets (.tres, .import, etc.) without failing audits.

Audit Result Mutual Exclusivity Rules:
- PASS: Critical paths aligned, scene graph topology matches 1:1, no type/instance shifts.
- REVIEW: Critical paths aligned, scene graph topology matches 1:1, BUT node renames or 1:1
          class/instance shifts occurred (e.g., KinematicBody2D -> CharacterBody2D).
- FAIL: Critical paths missing OR structural graph topology mismatch (missing/extra nodes, parent shifts).
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Matches [node ...] block declarations regardless of attribute order or multi-line spacing
_NODE_BLOCK_RE = re.compile(r"\[node\s+([^\]]+)\]", re.MULTILINE)

# Captures key-value pairs inside [node ...] attributes:
# - Group 1: Attribute name (e.g., 'name', 'type', 'parent', 'instance')
# - Group 2: Double-quoted string values (e.g., "Player")
# - Group 3: ExtResource expression calls (e.g., ExtResource("1_abc") or ExtResource( 1 ))
# - Group 4: Unquoted string values or numbers
_ATTR_RE = re.compile(r'(\w+)=(?:"([^"]*)"|(ExtResource\([^)]+\))|([^\s]+))')

# Critical file extensions that MUST match 1:1 for path alignment to pass
_CRITICAL_ARTEFACT_EXTS: Set[str] = {".gd", ".tscn"}

# Non-critical file extensions tracked as non-fatal informational warnings
_NON_CRITICAL_EXTS: Set[str] = {
    ".import",
    ".tres",
    ".gdshader",
    ".shader",
    ".png",
    ".wav",
    ".ogg",
    ".gdnlib",
    ".gdns",
}


@dataclass
class SceneDifference:
    """Detailed structural difference inside a scene (.tscn) file."""

    scene_path: str
    diff_type: str  # "node_count_mismatch", "missing_node", "extra_node", "parent_mismatch", "type_mismatch"
    details: str

    def to_dict(self) -> Dict[str, str]:
        """Converts instance to a JSON-serializable dictionary."""
        return {
            "scene_path": self.scene_path,
            "diff_type": self.diff_type,
            "details": self.details,
        }


@dataclass
class AuditResult:
    """Result of auditing a single project pair with critical failures and non-fatal warnings."""

    project_name: str
    passed: bool = False
    flagged_for_review: bool = False
    path_aligned: bool = True
    scene_isomorphic: bool = True
    missing_in_target: List[str] = field(default_factory=list)
    extra_in_target: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    scene_failures: List[str] = field(default_factory=list)
    granular_scene_diffs: List[SceneDifference] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the AuditResult instance to a JSON-serializable dictionary."""
        return {
            "project_name": self.project_name,
            "passed": self.passed,
            "flagged_for_review": self.flagged_for_review,
            "path_aligned": self.path_aligned,
            "scene_isomorphic": self.scene_isomorphic,
            "critical_path_differences": {
                "missing_in_target": self.missing_in_target,
                "extra_in_target": self.extra_in_target,
            },
            "warnings": self.warnings,
            "scene_differences": [diff.to_dict() for diff in self.granular_scene_diffs],
            "scene_failures": self.scene_failures,
            "failure_summary": self.notes,
        }


@dataclass
class SceneNode:
    """A single node extracted from a ``.tscn`` file with canonical pathing and topological signature."""

    name: str
    type: Optional[str] = None
    instance: Optional[str] = None
    parent: Optional[str] = None
    index: int = 0
    canonical_path: str = ""
    subtree_signature: str = ""  # AHU canonical parenthesis signature for topological matching

    @property
    def node_descriptor(self) -> str:
        """Returns string representation for diff logging (e.g., class type, instance path, or inherited)."""
        if self.type:
            return f"type:{self.type}"
        if self.instance:
            return f"instance:{self.instance}"
        return "type:Inherited/Scripted"


class IsomorphismAuditor:
    """Audits structural equivalence of Godot legacy (3.x) <-> target (4.x) project pairs."""

    def audit_pair(
        self, legacy_dir: Path, target_dir: Path, project_name: str = ""
    ) -> AuditResult:
        """Run full audit on a single project pair, resolving status strictly to PASS, REVIEW, or FAIL."""
        result = AuditResult(project_name=project_name or legacy_dir.name)

        # ------------------------------------------------------------------
        # 1. Critical Path Alignment Check (Case-Insensitive)
        # ------------------------------------------------------------------
        self._check_path_alignment(legacy_dir, target_dir, result)
        
        # MUTUAL EXCLUSIVITY RULE 1:
        # If critical files (.gd/.tscn) are missing in target, short-circuit immediately as FAIL.
        if not result.path_aligned:
            result.scene_isomorphic = False
            result.flagged_for_review = False
            result.passed = False
            if result.missing_in_target:
                result.notes.append(
                    f"Path Alignment Failure: {len(result.missing_in_target)} critical file(s) (.gd/.tscn) missing in target."
                )
            return result

        # ------------------------------------------------------------------
        # 2. Scene-Tree Graph Isomorphism Check
        # ------------------------------------------------------------------
        self._check_scene_isomorphism(legacy_dir, target_dir, result)

        # MUTUAL EXCLUSIVITY RULE 2:
        # Resolve status into exactly one mutually exclusive category: PASS, REVIEW, or FAIL.
        if not result.scene_isomorphic:
            # Structural failure (missing nodes, extra nodes, parent mismatches)
            result.passed = False
            result.flagged_for_review = False
            result.notes.append(
                f"Scene Isomorphism Failure: Found structural graph differences in {len(result.scene_failures)} scene file(s)."
            )
        elif result.flagged_for_review:
            # Graph topology matched 1:1, but class shifts or renames occurred
            result.passed = False
            result.notes.append(
                f"Flagged for Manual Review: Identical graph structures but 1:1 node type/instance shifts detected across scene(s)."
            )
        else:
            # Graph topology and node attributes matched cleanly 1:1
            result.passed = True

        return result

    def audit_all(self, project_pairs: List[Dict[str, Path]]) -> List[AuditResult]:
        """Audit every project pair and log summary execution statistics."""
        results: List[AuditResult] = []
        for pair in project_pairs:
            res = self.audit_pair(
                legacy_dir=pair["legacy"],
                target_dir=pair["target"],
                project_name=pair.get("name", ""),
            )
            status = "PASS" if res.passed else ("REVIEW" if res.flagged_for_review else "FAIL")
            logger.info("[%s] %s", status, res.project_name)
            results.append(res)

        passed = sum(1 for r in results if r.passed)
        review = sum(1 for r in results if r.flagged_for_review)
        failed = len(results) - passed - review
        logger.info(
            "Audit complete: %d passed, %d flagged for review, %d failed out of %d projects.",
            passed,
            review,
            failed,
            len(results),
        )
        return results

    # ------------------------------------------------------------------ #
    #  Path Alignment & Non-Critical Warning Tracking
    # ------------------------------------------------------------------ #

    def _check_path_alignment(
        self, legacy_dir: Path, target_dir: Path, result: AuditResult
    ) -> None:
        """Verifies presence of critical files (.gd/.tscn) and tracks missing/extra non-critical assets."""
        legacy_critical = self._collect_paths_case_insensitive(legacy_dir, _CRITICAL_ARTEFACT_EXTS)
        target_critical = self._collect_paths_case_insensitive(target_dir, _CRITICAL_ARTEFACT_EXTS)

        missing_critical = set(legacy_critical.keys()) - set(target_critical.keys())
        extra_critical = set(target_critical.keys()) - set(legacy_critical.keys())

        if missing_critical:
            result.path_aligned = False
            result.missing_in_target = sorted(str(legacy_critical[p]) for p in missing_critical)

        if extra_critical:
            result.extra_in_target = sorted(str(target_critical[p]) for p in extra_critical)

        # Check casing differences for matched critical files
        common_critical = set(legacy_critical.keys()) & set(target_critical.keys())
        for p in sorted(common_critical):
            legacy_path = legacy_critical[p]
            target_path = target_critical[p]
            if str(legacy_path) != str(target_path):
                result.warnings.append(
                    f"Casing mismatch in critical file: legacy '{legacy_path}' vs target '{target_path}'"
                )

        # Track non-critical files (.import, .tres, shaders, audio) as non-fatal warnings
        legacy_non_critical = self._collect_paths_case_insensitive(legacy_dir, _NON_CRITICAL_EXTS)
        target_non_critical = self._collect_paths_case_insensitive(target_dir, _NON_CRITICAL_EXTS)

        for p in sorted(set(legacy_non_critical.keys()) - set(target_non_critical.keys())):
            result.warnings.append(f"Non-critical file missing in target: {legacy_non_critical[p]}")

        for p in sorted(set(target_non_critical.keys()) - set(legacy_non_critical.keys())):
            result.warnings.append(f"Non-critical extra file present in target: {target_non_critical[p]}")

        # Check casing differences for matched non-critical files
        common_non_critical = set(legacy_non_critical.keys()) & set(target_non_critical.keys())
        for p in sorted(common_non_critical):
            legacy_path = legacy_non_critical[p]
            target_path = target_non_critical[p]
            if str(legacy_path) != str(target_path):
                result.warnings.append(
                    f"Casing mismatch in non-critical file: legacy '{legacy_path}' vs target '{target_path}'"
                )

    @staticmethod
    def _collect_paths_case_insensitive(root: Path, extensions: Set[str]) -> Dict[str, Path]:
        """Returns lowercased relative path string mapped to actual relative Path object."""
        path_map: Dict[str, Path] = {}
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                rel = p.relative_to(root)
                path_map[str(rel).lower()] = rel
        return path_map

    # ------------------------------------------------------------------ #
    #  Scene Parsing & Graph Normalization
    # ------------------------------------------------------------------ #

    def _check_scene_isomorphism(
        self, legacy_dir: Path, target_dir: Path, result: AuditResult
    ) -> None:
        """Parses and compares scene graphs across legacy and target projects."""
        legacy_tscns = self._collect_paths_case_insensitive(legacy_dir, {".tscn"})
        target_tscns = self._collect_paths_case_insensitive(target_dir, {".tscn"})

        has_isomorphism_failure = False
        has_review_flag = False

        for lower_rel, leg_rel in legacy_tscns.items():
            if lower_rel not in target_tscns:
                has_isomorphism_failure = True
                result.scene_failures.append(str(leg_rel))
                continue

            tar_rel = target_tscns[lower_rel]
            legacy_nodes = self._parse_scene_nodes(legacy_dir / leg_rel)
            target_nodes = self._parse_scene_nodes(target_dir / tar_rel)

            diffs, is_structural_failure, is_review = self._diff_scene_graphs(
                str(leg_rel), legacy_nodes, target_nodes
            )

            if diffs:
                result.granular_scene_diffs.extend(diffs)
                result.scene_failures.append(str(leg_rel))

            if is_structural_failure:
                has_isomorphism_failure = True
            elif is_review:
                has_review_flag = True

        # Priority resolution: Structural failures invalidate Review status
        if has_isomorphism_failure:
            result.scene_isomorphic = False
            result.flagged_for_review = False
        elif has_review_flag:
            result.scene_isomorphic = True
            result.flagged_for_review = True
        else:
            result.scene_isomorphic = True
            result.flagged_for_review = False

    @classmethod
    def _parse_scene_nodes(cls, tscn_path: Path) -> List[SceneNode]:
        """Extracts node records, resolves multi-line attributes, and calculates canonical paths."""
        text = tscn_path.read_text(errors="replace")
        nodes: List[SceneNode] = []
        path_mapping: Dict[str, str] = {}

        # Scan for [node ...] header blocks across single/multiple lines
        for idx, block_match in enumerate(_NODE_BLOCK_RE.finditer(text)):
            block_content = block_match.group(1)
            attrs = {}
            for m in _ATTR_RE.finditer(block_content):
                key = m.group(1)
                # Select group 2 (quoted), group 3 (ExtResource call), or group 4 (unquoted string)
                val = next(g for g in (m.group(2), m.group(3), m.group(4)) if g is not None)
                attrs[key] = val

            name = attrs.get("name", "")
            if not name:
                continue

            raw_parent = attrs.get("parent")
            node_type = attrs.get("type")
            node_instance = attrs.get("instance")

            # Canonical path resolution: convert '.', '', or relative parent paths into clean 'Root/Child' strings
            if raw_parent is None or raw_parent in ("", "."):
                canonical_path = name
            else:
                clean_parent = raw_parent.lstrip("./")
                parent_canonical = path_mapping.get(clean_parent, clean_parent)
                canonical_path = f"{parent_canonical}/{name}"

            normalized_parent = None if (raw_parent is None or raw_parent in ("", ".")) else raw_parent.lstrip("./")

            path_mapping[name] = canonical_path
            nodes.append(
                SceneNode(
                    name=name,
                    type=node_type,
                    instance=node_instance,
                    parent=normalized_parent,
                    index=idx,
                    canonical_path=canonical_path,
                )
            )

        # Compute AHU canonical subtree signatures for bottom-up topological matching
        cls._compute_subtree_signatures(nodes)
        return nodes

    @staticmethod
    def _compute_subtree_signatures(nodes: List[SceneNode]) -> None:
        """Computes AHU-style canonical parenthesis signatures for order-invariant graph matching.

        AHU Encoding Principle:
        - Leaf node signature: '()'
        - Subtree signature: '(' + sorted_child_signatures + ')'
        - Sorting child signatures lexicographically guarantees sibling order invariance.
        """
        children_map: Dict[Optional[str], List[SceneNode]] = {}
        for n in nodes:
            children_map.setdefault(n.parent, []).append(n)

        def build_sig(node: SceneNode) -> str:
            children = children_map.get(node.canonical_path, [])
            if not children:
                node.subtree_signature = "()"
                return "()"

            # Recursively compute and sort child signatures
            child_sigs = sorted(build_sig(c) for c in children)
            sig = "(" + "".join(child_sigs) + ")"
            node.subtree_signature = sig
            return sig

        # Compute signatures starting from root node(s)
        for root in children_map.get(None, []):
            build_sig(root)

    # ------------------------------------------------------------------ #
    #  Mutually Exclusive Graph Diffing Logic
    # ------------------------------------------------------------------ #

    def _diff_scene_graphs(
        self,
        scene_rel_path: str,
        legacy_nodes: List[SceneNode],
        target_nodes: List[SceneNode],
    ) -> Tuple[List[SceneDifference], bool, bool]:
        """Performs granular scene graph comparison, isolating structural failures from type/rename review shifts."""
        diffs: List[SceneDifference] = []

        # RULE 1: Any total node count difference is an immediate HARD FAIL (is_structural_failure=True, is_review_flag=False).
        if len(legacy_nodes) != len(target_nodes):
            diffs.append(
                SceneDifference(
                    scene_path=scene_rel_path,
                    diff_type="node_count_mismatch",
                    details=f"Legacy scene has {len(legacy_nodes)} nodes while target has {len(target_nodes)} nodes.",
                )
            )
            # Log missing/extra paths explicitly for reporting
            leg_paths = {n.canonical_path for n in legacy_nodes}
            tar_paths = {n.canonical_path for n in target_nodes}
            for path in sorted(leg_paths - tar_paths):
                diffs.append(SceneDifference(scene_rel_path, "missing_node", f"Node '{path}' missing in target."))
            for path in sorted(tar_paths - leg_paths):
                diffs.append(SceneDifference(scene_rel_path, "extra_node", f"Node '{path}' extra in target."))

            return diffs, True, False

        legacy_map = {n.canonical_path: n for n in legacy_nodes}
        target_map = {n.canonical_path: n for n in target_nodes}

        legacy_paths = set(legacy_map.keys())
        target_paths = set(target_map.keys())

        missing_paths = legacy_paths - target_paths
        extra_paths = target_paths - legacy_paths

        # Stage 1: Direct Path Alignment Check (If node paths match 1:1)
        if not missing_paths and not extra_paths:
            is_structural_failure = False
            type_mismatches = 0

            for path in sorted(legacy_paths):
                leg = legacy_map[path]
                tar = target_map[path]

                # Parent shifts alter graph topology -> Hard Structural Failure
                if leg.parent != tar.parent:
                    diffs.append(
                        SceneDifference(
                            scene_path=scene_rel_path,
                            diff_type="parent_mismatch",
                            details=f"Node '{path}' parent mismatch: legacy parent is '{leg.parent}', target parent is '{tar.parent}'.",
                        )
                    )
                    is_structural_failure = True

                # Class type or instance shifts -> Logged as type mismatch
                if (leg.type != tar.type) or (leg.instance != tar.instance):
                    type_mismatches += 1
                    diffs.append(
                        SceneDifference(
                            scene_path=scene_rel_path,
                            diff_type="type_mismatch",
                            details=f"Node '{path}' shift: legacy [{leg.node_descriptor}] <-> target [{tar.node_descriptor}].",
                        )
                    )

            is_review_flag = (not is_structural_failure) and (type_mismatches > 0)
            return diffs, is_structural_failure, is_review_flag

        # Stage 2: Order-Invariant Topology Matching (Handles Node Renames)
        mapping, topology_matched = self._find_tree_isomorphism_order_invariant(
            legacy_nodes, target_nodes
        )

        if topology_matched and mapping is not None:
            # Tree topology is 1:1 identical despite node renames! Report renames & type shifts
            type_mismatches = 0
            for leg, tar in mapping:
                if leg.name != tar.name:
                    diffs.append(
                        SceneDifference(
                            scene_path=scene_rel_path,
                            diff_type="type_mismatch",
                            details=f"Node renamed along invariant scene-tree path: '{leg.canonical_path}' -> '{tar.canonical_path}'.",
                        )
                    )
                if (leg.type != tar.type) or (leg.instance != tar.instance):
                    type_mismatches += 1
                    diffs.append(
                        SceneDifference(
                            scene_path=scene_rel_path,
                            diff_type="type_mismatch",
                            details=f"Node '{leg.canonical_path}' [{leg.node_descriptor}] shifted to '{tar.canonical_path}' [{tar.node_descriptor}].",
                        )
                    )

            return diffs, False, True

        # Stage 3: Graph topologies genuinely differ -> Hard Structural Failure
        for path in sorted(missing_paths):
            node = legacy_map[path]
            diffs.append(
                SceneDifference(
                    scene_path=scene_rel_path,
                    diff_type="missing_node",
                    details=f"Node '{path}' [{node.node_descriptor}] missing in target.",
                )
            )
        for path in sorted(extra_paths):
            node = target_map[path]
            diffs.append(
                SceneDifference(
                    scene_path=scene_rel_path,
                    diff_type="extra_node",
                    details=f"Node '{path}' [{node.node_descriptor}] extra in target.",
                )
            )

        return diffs, True, False

    def _find_tree_isomorphism_order_invariant(
        self, legacy_nodes: List[SceneNode], target_nodes: List[SceneNode]
    ) -> Tuple[Optional[List[Tuple[SceneNode, SceneNode]]], bool]:
        """Validates graph isomorphism using AHU subtree signatures without assuming any sibling order invariance."""
        if len(legacy_nodes) != len(target_nodes):
            return None, False

        def build_children_map(nodes: List[SceneNode]) -> Dict[Optional[str], List[SceneNode]]:
            children_map: Dict[Optional[str], List[SceneNode]] = {}
            for n in nodes:
                children_map.setdefault(n.parent, []).append(n)
            return children_map

        leg_children = build_children_map(legacy_nodes)
        tar_children = build_children_map(target_nodes)

        matched_pairs: List[Tuple[SceneNode, SceneNode]] = []

        def match_subtrees(leg_parent_path: Optional[str], tar_parent_path: Optional[str]) -> bool:
            leg_c = leg_children.get(leg_parent_path, [])
            tar_c = tar_children.get(tar_parent_path, [])

            if len(leg_c) != len(tar_c):
                return False

            unmatched_tar = list(tar_c)

            for leg_child in leg_c:
                best_match: Optional[SceneNode] = None

                # 1. First priority: match exact AHU signature AND node name (reordered sibling)
                for tar_child in unmatched_tar:
                    if (
                        leg_child.subtree_signature == tar_child.subtree_signature
                        and leg_child.name == tar_child.name
                    ):
                        best_match = tar_child
                        break

                # 2. Second priority: match exact AHU signature (renamed sibling)
                if best_match is None:
                    for tar_child in unmatched_tar:
                        if leg_child.subtree_signature == tar_child.subtree_signature:
                            best_match = tar_child
                            break

                if best_match is None:
                    return False  # No matching sibling topology found

                unmatched_tar.remove(best_match)
                matched_pairs.append((leg_child, best_match))

                # Recursively match subtrees down the tree
                if not match_subtrees(leg_child.canonical_path, best_match.canonical_path):
                    return False

            return True

        # Match recursively starting from root nodes (parent=None)
        success = match_subtrees(None, None)
        if success and len(matched_pairs) == len(legacy_nodes):
            return matched_pairs, True

        return None, False