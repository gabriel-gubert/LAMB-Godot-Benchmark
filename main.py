"""
Pipeline Orchestrator and Benchmark Runner for Godot GDScript Migration.

This orchestrator coordinates the complete end-to-end benchmark workflow:
1. Action 1 (action_audit_dataset):
   - Downloads legacy (3.x) and target (4.x) demo projects.
   - Executes graph-based isomorphism auditing on .tscn files and case-insensitive path checks.
   - Routes projects into mutually exclusive folders:
     * PASS -> dataset/audited_passed
     * REVIEW -> dataset/require_manual_review
     * FAIL -> Excluded from benchmark (detailed in audit_report.json)

2. Action 2 (action_run_baselines):
   - Copies full target project directory structures to output_dir upfront across all requested paradigms.
   - Executes automated migration paradigms (Zero-Shot, Hybrid RAG, LAMB) across project pairs.
   - Persists migrated project directory trees cleanly to disk in output_dir.

3. Action 3 (action_evaluate):
   - Runs static code analysis (AST edit distance, syntax verification, CodeBLEU).
   - Executes dynamic containerized runtime evaluations inside Docker harness.
   - Aggregates metrics into JSON reports and LaTeX benchmark tables.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Import project configuration and core functional modules
from config import *
from dataset.downloader import DatasetDownloader
from dataset.isomorphism_auditor import IsomorphismAuditor, AuditResult
from baselines.zero_shot import ZeroShotGenerator
from baselines.hybrid_rag import HybridRAGMigrator, LambChunkLoader
from baselines.standard_rag import StandardRAGMigrator
from baselines.lamb_runner import LambRunner
from verification.docker_harness import DockerHarness
from verification.static_analyzer import StaticAnalyzer
from verification.dynamic_runner import DynamicRunner

# Configure Rich Logger for formatted CLI output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("main")
console = Console()


# ----------------------------------------------------------------------
# Action 1: Dataset Acquisition & Mutually Exclusive Isomorphism Auditing
# ----------------------------------------------------------------------

def action_audit_dataset(
    dataset_dir: Path,
    force_download: bool = False,
    audit_report_path: Path = RESULTS_DIR / "audit_report.json",
) -> List[Dict[str, Any]]:
    """Downloads demo projects, performs structural isomorphism auditing, prints granular differences,
    exports a global report, copies passed projects to dataset/audited_passed, copies flagged projects
    to dataset/require_manual_review, and ships individual audit_report.json files to each project folder.
    """
    console.print("\n[bold blue]=== Step 1: Downloading & Auditing Dataset ===[/bold blue]")

    # 1. Download/discover legacy <-> target repository pairs
    downloader = DatasetDownloader(dataset_dir=dataset_dir)
    downloader.download(force=force_download)
    pairs = downloader.discover_project_pairs()

    # 2. Run structural isomorphism auditor across all repository pairs
    auditor = IsomorphismAuditor()
    audit_results: List[AuditResult] = auditor.audit_all(pairs)

    valid_pairs: List[Dict[str, Any]] = []
    full_audit_report: List[Dict[str, Any]] = []

    # Define clean destination root folders for routed projects
    audited_passed_dir = dataset_dir / "audited_passed"
    review_dir = dataset_dir / "require_manual_review"

    # Wipe existing audited directories if force re-download is requested
    if force_download:
        if audited_passed_dir.exists():
            shutil.rmtree(audited_passed_dir)
        if review_dir.exists():
            shutil.rmtree(review_dir)

    audited_passed_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    # 3. Process audit results and route projects into mutually exclusive destinations
    for pair, res in zip(pairs, audit_results):
        res_dict = res.to_dict()
        full_audit_report.append(res_dict)

        # ------------------------------------------------------------------
        # Mutually Exclusive Category Routing: PASS, REVIEW, or FAIL
        # ------------------------------------------------------------------

        if res.passed:
            # CATEGORY 1: PASS -> Copy project pair to dataset/audited_passed
            target_base_dir = audited_passed_dir
            audited_legacy_dir = target_base_dir / "legacy" / pair["name"]
            audited_target_dir = target_base_dir / "target" / pair["name"]

            if audited_legacy_dir.exists():
                shutil.rmtree(audited_legacy_dir)
            if audited_target_dir.exists():
                shutil.rmtree(audited_target_dir)

            shutil.copytree(pair["legacy"], audited_legacy_dir)
            shutil.copytree(pair["target"], audited_target_dir)

            # Write individual project audit report JSON to project root folders
            indiv_report_json = json.dumps(res_dict, indent=2)
            (audited_legacy_dir / "audit_report.json").write_text(indiv_report_json)
            (audited_target_dir / "audit_report.json").write_text(indiv_report_json)

            # Discover and collect valid matching .gd script pairs
            for gd_file in audited_legacy_dir.rglob("*.gd"):
                rel_path = gd_file.relative_to(audited_legacy_dir)
                target_gd = audited_target_dir / rel_path
                if target_gd.exists():
                    valid_pairs.append({
                        "project_name": pair["name"],
                        "legacy_project_dir": audited_legacy_dir,
                        "target_project_dir": audited_target_dir,
                        "relative_script_path": rel_path,
                        "legacy_script_path": gd_file,
                        "target_script_path": target_gd,
                        "flagged_for_review": False,
                    })

            console.print(
                f"\n[bold green]✔ Project Passed Audit (Copied to dataset/audited_passed):[/bold green] [cyan]{res.project_name}[/cyan] "
                f"([yellow]{len(res.warnings)} Non-Fatal Warning(s)[/yellow])"
            )
            for warn in res.warnings[:5]:
                console.print(f"  [yellow]• Warning:[/yellow] {warn}")
            if len(res.warnings) > 5:
                console.print(f"  [dim]... and {len(res.warnings) - 5} additional warnings.[/dim]")

        elif res.flagged_for_review:
            # CATEGORY 2: REVIEW -> Copy project pair to dataset/require_manual_review
            target_base_dir = review_dir
            audited_legacy_dir = target_base_dir / "legacy" / pair["name"]
            audited_target_dir = target_base_dir / "target" / pair["name"]

            if audited_legacy_dir.exists():
                shutil.rmtree(audited_legacy_dir)
            if audited_target_dir.exists():
                shutil.rmtree(audited_target_dir)

            shutil.copytree(pair["legacy"], audited_legacy_dir)
            shutil.copytree(pair["target"], audited_target_dir)

            # Write individual project audit report JSON
            indiv_report_json = json.dumps(res_dict, indent=2)
            (audited_legacy_dir / "audit_report.json").write_text(indiv_report_json)
            (audited_target_dir / "audit_report.json").write_text(indiv_report_json)

            for gd_file in audited_legacy_dir.rglob("*.gd"):
                rel_path = gd_file.relative_to(audited_legacy_dir)
                target_gd = audited_target_dir / rel_path
                if target_gd.exists():
                    valid_pairs.append({
                        "project_name": pair["name"],
                        "legacy_project_dir": audited_legacy_dir,
                        "target_project_dir": audited_target_dir,
                        "relative_script_path": rel_path,
                        "legacy_script_path": gd_file,
                        "target_script_path": target_gd,
                        "flagged_for_review": True,
                    })

            console.print(
                f"\n[bold yellow]⚠ Project Flagged for Review (Moved to dataset/require_manual_review):[/bold yellow] [cyan]{res.project_name}[/cyan] "
                f"([yellow]1:1 Node Type Shifts / Renames Detected[/yellow])"
            )
            for diff in res.granular_scene_diffs[:5]:
                console.print(f"  [yellow]• Shift:[/yellow] [{diff.scene_path}] {diff.details}")
            if len(res.granular_scene_diffs) > 5:
                console.print(f"  [dim]... and {len(res.granular_scene_diffs) - 5} additional shifts.[/dim]")

        else:
            # CATEGORY 3: FAIL -> Structural / Path Error (Project excluded from dataset)
            total_missing = len(res.missing_in_target)
            total_extra = len(res.extra_in_target)
            total_scene_diffs = len(res.granular_scene_diffs)
            total_warnings = len(res.warnings)

            console.print(
                f"\n[bold red]✖ Project Failed Audit:[/bold red] [cyan]{res.project_name}[/cyan]\n"
                f"  [bold red]Absolute Mismatch Totals:[/bold red] "
                f"[yellow]{total_missing}[/yellow] Missing Critical File(s), "
                f"[yellow]{total_extra}[/yellow] Extra File(s), "
                f"[yellow]{total_scene_diffs}[/yellow] Scene Graph Difference(s), "
                f"[yellow]{total_warnings}[/yellow] Non-Fatal Warning(s)"
            )

            for summary in res.notes:
                console.print(f"  [bold yellow]• Audit Summary:[/bold yellow] {summary}")

            if res.missing_in_target:
                console.print(f"  [yellow]• Missing Critical Files (.gd/.tscn) [Total: {total_missing}]:[/yellow]")
                for missing in res.missing_in_target[:5]:
                    console.print(f"    - [dim]{missing}[/dim]")
                if total_missing > 5:
                    console.print(f"    - [dim]... and {total_missing - 5} more missing critical files.[/dim]")

            if res.extra_in_target:
                console.print(f"  [yellow]• Extra Files in Target [Total: {total_extra}]:[/yellow]")
                for extra in res.extra_in_target[:5]:
                    console.print(f"    - [dim]{extra}[/dim]")
                if total_extra > 5:
                    console.print(f"    - [dim]... and {total_extra - 5} more extra files.[/dim]")

            if res.granular_scene_diffs:
                console.print(f"  [yellow]• Granular Scene Graph Node Differences [Total: {total_scene_diffs}]:[/yellow]")
                for diff in res.granular_scene_diffs[:8]:
                    console.print(f"    - [{diff.scene_path}] ({diff.diff_type}): {diff.details}")
                if total_scene_diffs > 8:
                    console.print(f"    - [dim]... and {total_scene_diffs - 8} more node differences.[/dim]")

            if res.warnings:
                console.print(f"  [yellow]• Non-Fatal Asset Warnings [Total: {total_warnings}]:[/yellow]")
                for warn in res.warnings[:5]:
                    console.print(f"    - {warn}")
                if total_warnings > 5:
                    console.print(f"    - [dim]... and {total_warnings - 5} more asset warnings.[/dim]")

    # 4. Write aggregate audit summary JSON report for paper benchmarking
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report_path.write_text(json.dumps(full_audit_report, indent=2))
    console.print(f"\n[bold green]Global aggregate audit report written to {audit_report_path}[/bold green]")

    # Print pipeline totals summary
    passed_projects = sum(1 for res in audit_results if res.passed)
    review_projects = sum(1 for res in audit_results if res.flagged_for_review)
    failed_projects = sum(1 for res in audit_results if not res.passed and not res.flagged_for_review)

    passed_script_pairs = sum(1 for p in valid_pairs if not p.get("flagged_for_review"))
    review_script_pairs = sum(1 for p in valid_pairs if p.get("flagged_for_review"))

    console.print(
        f"[bold green]Audit Summary:[bold green] Retained [cyan]{passed_projects}[/cyan] passed projects "
        f"([dim]{passed_script_pairs} script pairs[/dim]) in [dim]audited_passed/[/dim] and "
        f"[yellow]{review_projects}[/yellow] projects ([dim]{review_script_pairs} script pairs[/dim]) "
        f"in [dim]require_manual_review/[/dim]. [red]{failed_projects}[/red] projects failed."
    )
    return valid_pairs

def load_existing_audited_pairs(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Loads existing audited project script pairs directly from disk without re-running the auditor."""
    valid_pairs: List[Dict[str, Any]] = []
    
    for base_folder in ["audited_passed", "require_manual_review"]:
        base_dir = dataset_dir / base_folder
        legacy_base = base_dir / "legacy"
        target_base = base_dir / "target"
        
        if not legacy_base.exists() or not target_base.exists():
            continue

        is_review = (base_folder == "require_manual_review")

        # Exclude review projects if configured to run passed projects only
        if is_review and not REQUIRE_REVIEW_EVALUATION:
            continue

        legacy_projects: Dict[Path, Path] = {}

        for godot_file in legacy_base.rglob("project.godot"):
            legacy_project_dir = godot_file.parent
            legacy_rel = legacy_project_dir.relative_to(legacy_base)
            legacy_projects[legacy_rel] = legacy_project_dir

        for legacy_proj_name, legacy_proj_dir in legacy_projects.items():
            if not legacy_proj_dir.is_dir():
                continue

            proj_name = legacy_proj_name
            target_proj_dir = target_base / proj_name

            if not target_proj_dir.exists():
                continue

            # Collect matching GDScript pairs
            for legacy_gd_file in legacy_proj_dir.rglob("*.gd"):
                rel_path = legacy_gd_file.relative_to(legacy_proj_dir)
                target_gd_file = target_proj_dir / rel_path

                if target_gd_file.exists():
                    valid_pairs.append({
                        "project_name": str(proj_name),
                        "legacy_project_dir": legacy_proj_dir,
                        "target_project_dir": target_proj_dir,
                        "relative_script_path": rel_path,
                        "legacy_script_path": legacy_gd_file,
                        "target_script_path": target_gd_file,
                        "flagged_for_review": is_review,
                    })

    return valid_pairs

# ----------------------------------------------------------------------
# Action 2: Baseline & Framework Migration Execution
# ----------------------------------------------------------------------

def action_run_baselines(
    paradigms: List[str],
    valid_pairs: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Executes selected migration paradigms over full project copies persisted to disk."""

    console.print("\n[bold blue]=== Step 2: Running Migration Baselines (Decoupled) ===[/bold blue]")
    migration_records: List[Dict[str, Any]] = []

    # 1. Pre-copy all target projects to output directory for each paradigm upfront
    console.print("[bold cyan]Copying target project structures to output directory...[/bold cyan]")
    unique_projects: Dict[str, Path] = {}
    for item in valid_pairs:
        unique_projects[item["project_name"]] = item["target_project_dir"]

    for p_name in paradigms:
        for proj_name, target_dir in unique_projects.items():
            migrated_proj_dir = output_dir / p_name / proj_name
            if not migrated_proj_dir.exists():
                logger.info("Copying target project '%s' to %s", proj_name, migrated_proj_dir)
                shutil.copytree(target_dir, migrated_proj_dir)

    # 2. Instantiate selected migration engines
    zero_shot_gen = ZeroShotGenerator() if "zero-shot" in paradigms else None
    standard_rag_migrator = StandardRAGMigrator() if "standard-rag" in paradigms else None
    rag_migrator = HybridRAGMigrator() if "hybrid-rag" in paradigms else None
    lamb_runner = LambRunner(
        LAMB_MAPPING_ALIAS,
        LAMB_MAX_ATTEMPTS,
        LAMB_SUMMARIZATION_THRESHOLD,
        LAMB_VERBOSE
    ) if "lamb" in paradigms else None

    if "lamb" in paradigms and lamb_runner:
        lamb_runner.list_config(LAMB_CONFIG_PATH)

    # Load LAMB's exported unbundled unique Godot 4 (v2) chunks into VectorDB/BM25 for Hybrid-RAG
    if rag_migrator or standard_rag_migrator:
        docs_godot4 = DOCS_DIR / "godot4-docs.json"

        if docs_godot4.exists():
            logger.info("Loading unbundled unique Godot 4 chunks for Hybrid RAG from %s...", docs_godot4)
            godot_docs = LambChunkLoader.load_from_json(docs_godot4)
            if godot_docs:
                if rag_migrator:
                    rag_migrator.index_documents(godot_docs)
                if standard_rag_migrator:
                    standard_rag_migrator.index_documents(godot_docs)
            else:
                logger.warning("No valid chunks parsed from %s", docs_godot4)
        else:
            logger.warning(
                "Could not find %s! Ensure LAMB mapping has run and exported unique chunks.",
                docs_godot4,
            )

    # 3. Process each script pair across requested paradigms
    for item in valid_pairs:
        rel_path = item["relative_script_path"]
        proj_name = item["project_name"]

        for p_name in paradigms:
            migrated_proj_dir = output_dir / p_name / proj_name
            out_file = migrated_proj_dir / rel_path

            logger.info("Migrating [%s] %s/%s", p_name, proj_name, rel_path)
            res_data: Dict[str, Any] = {}

            # Execute migration generator based on paradigm
            if p_name == "zero-shot" and zero_shot_gen:
                res_data = zero_shot_gen.migrate_file(
                    legacy_path=item["legacy_script_path"],
                    output_path=out_file,
                )
            elif p_name == "standard-rag" and standard_rag_migrator:
                res_data = standard_rag_migrator.migrate_file(
                    legacy_path=item["legacy_script_path"],
                    output_path=out_file,
                )
            elif p_name == "hybrid-rag" and rag_migrator:
                res_data = rag_migrator.migrate_file(
                    legacy_path=item["legacy_script_path"],
                    output_path=out_file,
                )
            elif p_name == "lamb" and lamb_runner:
                report_file = out_file.with_suffix(".json")
                res_data = lamb_runner.migrate_file(
                    legacy_path=item["legacy_script_path"],
                    output_path=out_file,
                    report_path=report_file,
                )

            # Record migration output metadata
            migration_records.append({
                "paradigm": p_name,
                "project_name": proj_name,
                "relative_script_path": str(rel_path),
                "migrated_file_path": out_file,
                "migrated_project_dir": migrated_proj_dir,
                "target_script_path": item["target_script_path"],
                "target_project_dir": item["target_project_dir"],
                "latency": res_data.get("latency", 0.0),
                "symbols_extracted": res_data.get("symbols_extracted", []),
                "retrieved_chunk_hashes": res_data.get("retrieved_chunk_hashes", []),
            })

    # Export audit JSON if hybrid-rag was executed
    rag_records = [r for r in migration_records if r["paradigm"] == "hybrid-rag"]
    standard_rag_records = [r for r in migration_records if r["paradigm"] == "standard-rag"]
    if rag_records or standard_rag_records:
        if standard_rag_records:
            standard_rag_audit_report = RESULTS_DIR / "standard_rag_context_audit.json"
            standard_rag_audit_report.parent.mkdir(parents=True, exist_ok=True)
            standard_rag_audit_data = [
                {
                    "project_name": rec["project_name"],
                    "script": rec["relative_script_path"],
                    "paradigm": rec["paradigm"],
                    "symbols_extracted": rec["symbols_extracted"],
                    "retrieved_chunk_hashes": rec["retrieved_chunk_hashes"],
                }
                for rec in standard_rag_records
            ]
            standard_rag_audit_report.write_text(json.dumps(standard_rag_audit_data, indent=2, default=str))
            console.print(f"[bold green]Standard RAG Chunk & Symbol Audit Report saved to {standard_rag_audit_report}[/bold green]")
        if rag_records:
            rag_audit_report = RESULTS_DIR / "hybrid_rag_context_audit.json"
            rag_audit_report.parent.mkdir(parents=True, exist_ok=True)
            rag_audit_data = [
                {
                    "project_name": rec["project_name"],
                    "script": rec["relative_script_path"],
                    "paradigm": rec["paradigm"],
                    "symbols_extracted": rec["symbols_extracted"],
                    "retrieved_chunk_hashes": rec["retrieved_chunk_hashes"],
                }
                for rec in rag_records
            ]
            rag_audit_report.write_text(json.dumps(rag_audit_data, indent=2, default=str))
            console.print(f"[bold green]Hybrid RAG Chunk & Symbol Audit Report saved to {rag_audit_report}[/bold green]")

    return migration_records


# ----------------------------------------------------------------------
# Action 3: Containerized Evaluation & Metrics Aggregation
# ----------------------------------------------------------------------

def action_evaluate(
    records: List[Dict[str, Any]],
    results_dir: Path,
    report_filename: str = "experiment_results.json",
) -> Dict[str, Any]:
    """Runs static and dynamic validation harness directly against persisted output project directories."""
    console.print("\n[bold blue]=== Step 3: Containerized Evaluation & Metrics ===[/bold blue]")

    harness = DockerHarness()
    static_analyzer = StaticAnalyzer()
    dynamic_runner = DynamicRunner(docker_harness=harness)

    paradigm_metrics: Dict[str, Dict[str, Any]] = {}
    detailed_records: List[Dict[str, Any]] = []

    for rec in records:
        p_name = rec["paradigm"]
        proj_name = rec["project_name"]
        rel_script_path = rec["relative_script_path"]

        if p_name not in paradigm_metrics:
            paradigm_metrics[p_name] = {
                "total": 0,
                "syntax_errors": 0,
                "ast_distances": [],
                "codebleu_scores": [],
                "passes": 0,
                "latencies": [],
            }

        migrated_code = rec["migrated_file_path"].read_text(errors="replace")
        reference_code = rec["target_script_path"].read_text(errors="replace")

        # 1. Static Analysis (Syntax check, AST Distance, CodeBLEU)
        container_rel_path = rel_script_path

        static_res = static_analyzer.analyse_file(
            generated_code=migrated_code,
            reference_code=reference_code,
            docker_harness=harness,
            container_script_path=str(container_rel_path),
            project_dir=str(rec["migrated_project_dir"]),
        )

        # 2. Dynamic Runtime Evaluation on the persisted project directory in Docker
        dynamic_res = dynamic_runner.evaluate_runtime(
            migrated_project_dir=str(rec["migrated_project_dir"]),
        )

        syntax_passed = static_res.get("syntax_passed", True)
        ast_dist = static_res.get("ast_edit_distance", 0)
        codebleu = static_res.get("codebleu", 0.0)
        dynamic_passed = dynamic_res.get("passed", False)
        latency = rec.get("latency", 0.0)

        stats = paradigm_metrics[p_name]
        stats["total"] += 1
        if not syntax_passed:
            stats["syntax_errors"] += 1
        stats["ast_distances"].append(ast_dist)
        stats["codebleu_scores"].append(codebleu)
        if dynamic_passed:
            stats["passes"] += 1
        stats["latencies"].append(latency)

        detailed_records.append({
            "paradigm": p_name,
            "project_name": proj_name,
            "script_path": rel_script_path,
            "syntax_passed": syntax_passed,
            "ast_distance": ast_dist,
            "codebleu": codebleu,
            "dynamic_passed": dynamic_passed,
            "latency": latency,
        })

    # Render Detailed Per-Demo Visualisation Table to Console
    demo_table = Table(title="Per-Demo Project Migration Metrics", show_lines=True)
    demo_table.add_column("Paradigm", style="cyan", no_wrap=True)
    demo_table.add_column("Project / Script Path", style="white")
    demo_table.add_column("Syntax", style="bold")
    demo_table.add_column("AST Dist", style="yellow")
    demo_table.add_column("CodeBLEU", style="green")
    demo_table.add_column("Dynamic Status", style="bold")
    demo_table.add_column("Latency (s)", style="blue")

    for d in detailed_records:
        syn_str = "[green]PASS[/green]" if d["syntax_passed"] else "[red]FAIL[/red]"
        dyn_str = "[bold green]PASS[/bold green]" if d["dynamic_passed"] else "[bold red]FAIL[/bold red]"
        demo_table.add_row(
            d["paradigm"],
            f"{d['project_name']}\n[dim]{d['script_path']}[/dim]",
            syn_str,
            str(d["ast_distance"]),
            f"{d['codebleu']:.4f}",
            dyn_str,
            f"{d['latency']:.2f}s",
        )

    console.print("\n", demo_table)

    # Compute and Render Aggregated Paradigm Benchmark Table
    summary: Dict[str, Any] = {"aggregated": {}, "per_demo": detailed_records}
    table = Table(title="Godot Code Migration Evaluation Benchmark (Summary)")
    table.add_column("Migration Paradigm", style="cyan", no_wrap=True)
    table.add_column("Syntax Error Rate (%)", style="magenta")
    table.add_column("Mean AST Distance", style="yellow")
    table.add_column("CodeBLEU", style="green")
    table.add_column("Dynamic Pass Rate (%)", style="bold green")
    table.add_column("Avg Latency (s)", style="blue")

    for p_name, stats in paradigm_metrics.items():
        tot = stats["total"] or 1
        syn_rate = (stats["syntax_errors"] / tot) * 100.0
        mean_ast = sum(stats["ast_distances"]) / tot
        mean_codebleu = sum(stats["codebleu_scores"]) / tot
        pass_rate = (stats["passes"] / tot) * 100.0
        avg_lat = sum(stats["latencies"]) / tot

        summary["aggregated"][p_name] = {
            "syntax_error_rate": round(syn_rate, 2),
            "mean_ast_distance": round(mean_ast, 2),
            "codebleu": round(mean_codebleu, 4),
            "pass_rate": round(pass_rate, 2),
            "avg_latency": round(avg_lat, 2),
        }

        table.add_row(
            p_name,
            f"{syn_rate:.2f}%",
            f"{mean_ast:.2f}",
            f"{mean_codebleu:.4f}",
            f"{pass_rate:.2f}%",
            f"{avg_lat:.2f}s",
        )

    console.print("\n", table)

    # Write summary results to JSON
    report_file = results_dir / report_filename
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(summary, indent=2))
    console.print(f"\n[bold green]Results written to {report_file}[/bold green]")

    # Generate and write LaTeX formatted benchmark tables for publication
    tex_summary_path = results_dir / "summary_table.tex"
    tex_demo_path = results_dir / "per_demo_table.tex"

    # Aggregated Summary LaTeX Table
    tex_summary_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Evaluation Summary Across Migration Paradigms}",
        "\\label{tab:migration_summary}",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "\\textbf{Paradigm} & \\textbf{Syntax Error (\\%)} & \\textbf{Mean AST Dist} & \\textbf{CodeBLEU} & \\textbf{Pass Rate (\\%)} & \\textbf{Avg Latency (s)} \\\\",
        "\\hline",
    ]
    for p_name, metrics in summary["aggregated"].items():
        tex_summary_lines.append(
            f"{p_name} & {metrics['syntax_error_rate']:.2f} & {metrics['mean_ast_distance']:.2f} & {metrics['codebleu']:.4f} & {metrics['pass_rate']:.2f} & {metrics['avg_latency']:.2f} \\\\"
        )
    tex_summary_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])
    tex_summary_path.write_text("\n".join(tex_summary_lines))

    # Per-Demo Detailed LaTeX Table
    tex_demo_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Detailed Per-Demo Project Migration Metrics}",
        "\\label{tab:per_demo_metrics}",
        "\\begin{tabular}{llccccr}",
        "\\hline",
        "\\textbf{Paradigm} & \\textbf{Project / Script} & \\textbf{Syntax} & \\textbf{AST Dist} & \\textbf{CodeBLEU} & \\textbf{Dynamic} & \\textbf{Latency (s)} \\\\",
        "\\hline",
    ]
    for d in detailed_records:
        clean_proj = d["project_name"].replace("_", "\\_")
        clean_path = d["script_path"].replace("_", "\\_")
        syn_tex = "PASS" if d["syntax_passed"] else "FAIL"
        dyn_tex = "PASS" if d["dynamic_passed"] else "FAIL"
        tex_demo_lines.append(
            f"{d['paradigm']} & {clean_proj} ({clean_path}) & {syn_tex} & {d['ast_distance']} & {d['codebleu']:.4f} & {dyn_tex} & {d['latency']:.2f} \\\\"
        )
    tex_demo_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])
    tex_demo_path.write_text("\n".join(tex_demo_lines))

    console.print(f"[bold green]LaTeX summary table exported to {tex_summary_path}[/bold green]")
    console.print(f"[bold green]LaTeX per-demo table exported to {tex_demo_path}[/bold green]\n")

    return summary


# ----------------------------------------------------------------------
# CLI Main Interface
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Godot 3.x to 4.x Migration Benchmark Pipeline"
    )
    parser.add_argument(
        "--audit-dataset",
        action="store_true",
        help="Download repositories and execute structural isomorphism auditing.",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Execute code migration across paradigms.",
    )
    parser.add_argument(
        "--paradigm",
        choices=["zero-shot", "standard-rag", "hybrid-rag", "lamb", "all"],
        default="all",
        help="Specify migration paradigm to execute.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run containerized static analysis and dynamic runtime evaluation.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading project repositories.",
    )
    parser.add_argument(
        "--report-output",
        default="experiment_results.json",
        help="Output JSON filename for evaluation metrics summary.",
    )
    parser.add_argument(
        "--audit-report",
        default="audit_report.json",
        help="Output JSON filename for dataset audit report.",
    )

    args = parser.parse_args()

    # Default execution mode: run full pipeline if no specific stage flag is set
    run_all = not (args.audit_dataset or args.run_baselines or args.evaluate)

    valid_pairs = []
    audit_report_file = RESULTS_DIR / args.audit_report

    # Step 1: Execute or load dataset audit
    if args.audit_dataset or run_all:
        valid_pairs = action_audit_dataset(
            dataset_dir=DATASET_DIR,
            force_download=args.force_download,
            audit_report_path=audit_report_file,
        )
    else:
        console.print("\n[bold yellow]Skipping audit dataset step. Loading existing audited projects from disk...[/bold yellow]")
        valid_pairs = load_existing_audited_pairs(dataset_dir=DATASET_DIR)
        console.print(f"[bold green]Loaded {len(valid_pairs)} script pairs from disk.[/bold green]")

    # Step 2 & 3: Baseline execution & Containerized evaluation
    migration_records = []
    if args.run_baselines or args.evaluate or run_all:
        selected_paradigms = (
            ["zero-shot", "standard-rag", "hybrid-rag", "lamb"]
            if args.paradigm == "all"
            else [args.paradigm]
        )
        if args.run_baselines or run_all:
            migration_records = action_run_baselines(
                paradigms=selected_paradigms,
                valid_pairs=valid_pairs,
                output_dir=OUTPUT_DIR,
            )
        else:
            # Reconstruct migration records from existing output directories for standalone evaluation
            for p_name in selected_paradigms:
                for item in valid_pairs:
                    migrated_proj_dir = OUTPUT_DIR / p_name / item["project_name"]
                    out_file = migrated_proj_dir / item["relative_script_path"]
                    if out_file.exists():
                        migration_records.append({
                            "paradigm": p_name,
                            "project_name": item["project_name"],
                            "relative_script_path": str(item["relative_script_path"]),
                            "migrated_file_path": out_file,
                            "migrated_project_dir": migrated_proj_dir,
                            "target_script_path": item["target_script_path"],
                            "target_project_dir": item["target_project_dir"],
                            "latency": 0.0,
                        })

    if args.evaluate or run_all:
        action_evaluate(
            records=migration_records,
            results_dir=RESULTS_DIR,
            report_filename=args.report_output,
        )


if __name__ == "__main__":
    main()