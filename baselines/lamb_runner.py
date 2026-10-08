"""Proposed Approach: LAMB CLI Runner.

Wraps the ``lamb`` executable to perform GDScript migration via
``lamb migrate`` CLI commands.
"""

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LAMB_MAPPING_ALIAS

logger = logging.getLogger(__name__)


class LambRunner:
    """Runs the LAMB CLI for Godot 3 -> 4 GDScript migration aligned with ConfigSchema."""

    def __init__(
        self,
        mapping_alias: str,
        max_attempts: Optional[int] = None,
        summarization_threshold: Optional[int] = None,
        verbose: Optional[bool] = None
    ):
        self.max_attempts = max_attempts
        self.mapping_alias = mapping_alias
        self.summarization_threshold = summarization_threshold
        self.verbose = verbose

        # Verify lamb is available
        if not shutil.which("lamb"):
            logger.warning(
                "'%s' not found on PATH. Migration calls will fail.",
                "lamb",
            )

    # ---------------------------------------------------------------- #
    #  Public API
    # ---------------------------------------------------------------- #

    def migrate_file(
        self,
        legacy_path: Path,
        output_path: Path,
        report_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Migrate a single GDScript file via ``lamb migrate``.

        Parameters
        ----------
        legacy_path : Path
            Path to the Godot 3.x ``.gd`` file.
        output_path : Path
            Destination for the migrated file.
        report_path : Path, optional
            Where LAMB should write its JSON report.

        Returns
        -------
        dict
            Contains ``generated_code``, ``latency``, ``returncode``,
            ``report`` (parsed JSON if available), and ``stderr``.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(legacy_path, output_path, report_path)
        logger.info("Running: %s", " ".join(cmd))

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        latency = time.perf_counter() - t0

        generated_code = ""
        if output_path.exists():
            generated_code = output_path.read_text(errors="replace")

        report_data: Dict[str, Any] = {}
        if report_path and report_path.exists():
            try:
                report_data = json.loads(report_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Could not parse LAMB report: %s", report_path)

        if proc.returncode != 0:
            logger.error(
                "lamb migrate failed (rc=%d): %s",
                proc.returncode,
                proc.stderr.strip(),
            )

        return {
            "generated_code": generated_code,
            "latency": latency,
            "returncode": proc.returncode,
            "report": report_data,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "confidence": report_data.get("confidence"),
            "tokens": report_data.get("token_usage", {}),
            "audit_records": report_data.get("audit", []),
        }

    def list_config(self, config_path: Path) -> List[str]:
        """Construct the ``lamb config --list`` CLI command list."""

        config_proc = subprocess.run(
            ["lamb", "config", "--list"], capture_output=True, text=True, timeout=60
        )

        config_path.write_text(config_proc.stdout, encoding="utf-8")

    # ---------------------------------------------------------------- #
    #  Internal helpers
    # ---------------------------------------------------------------- #

    def _build_command(
        self,
        legacy_path: Path,
        output_path: Path,
        report_path: Optional[Path],
    ) -> List[str]:
        """Construct the ``lamb migrate`` CLI command list."""
        cmd = ["lamb"]

        # Command Name
        cmd.append("migrate")

        # Migrate Command Arguments
        cmd.extend(["--file", str(legacy_path)])
        cmd.extend(["--output", str(output_path)])

        if isinstance(self.mapping_alias, str):
            cmd.extend(["--alias", self.mapping_alias])

        # Config Overrides
        if isinstance(self.max_attempts, int):
            cmd.extend(["--max-attempts", str(self.max_attempts)])

        if isinstance(self.summarization_threshold, int):
            cmd.extend(["--summarization-threshold", str(self.summarization_threshold)])

        if isinstance(report_path, Path):
            cmd.extend(["--report", str(report_path)])

        # Global Config Flags
        if isinstance(self.verbose, bool) and self.verbose:
            cmd.append("--verbose")

        return cmd