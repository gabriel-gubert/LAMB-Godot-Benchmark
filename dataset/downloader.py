"""Repository Downloader for Godot demo projects.

Clones official Godot demo project repositories (3.x legacy and 4.x target
branches) and maps legacy projects to their corresponding target versions.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DATASET_DIR, DEMO_REPO_URL, LEGACY_BRANCH, TARGET_BRANCH

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Downloads and organises Godot demo project pairs (legacy ↔ target)."""

    def __init__(
        self,
        dataset_dir: Optional[Path] = None,
        repo_url: str = DEMO_REPO_URL,
        legacy_branch: str = LEGACY_BRANCH,
        target_branch: str = TARGET_BRANCH,
    ) -> None:
        self.dataset_dir = dataset_dir or DATASET_DIR
        self.repo_url = repo_url
        self.legacy_branch = legacy_branch
        self.target_branch = target_branch

        self.legacy_dir = self.dataset_dir / "legacy"
        self.target_dir = self.dataset_dir / "target"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def download(self, force: bool = False) -> Tuple[Path, Path]:
        """Clone both branches.  Returns *(legacy_dir, target_dir)*.

        Parameters
        ----------
        force : bool
            If *True*, delete existing clones before re-downloading.
        """
        if force:
            for d in (self.legacy_dir, self.target_dir):
                if d.exists():
                    logger.info("Removing existing clone: %s", d)
                    shutil.rmtree(d)

        self._clone_branch(self.legacy_branch, self.legacy_dir)
        self._clone_branch(self.target_branch, self.target_dir)

        return self.legacy_dir, self.target_dir

    def discover_project_pairs(self) -> List[Dict[str, Path]]:
        """Return a list of ``{legacy: Path, target: Path}`` dicts.

        A *project* is any directory inside the clone root that contains a
        ``project.godot`` file.  Pairs are matched by identical relative
        paths from the repository root.
        """
        legacy_projects = self._find_projects(self.legacy_dir)
        target_projects = self._find_projects(self.target_dir)

        pairs: List[Dict[str, Path]] = []
        for rel_path in sorted(legacy_projects):
            if rel_path in target_projects:
                pairs.append(
                    {
                        "name": str(rel_path),
                        "legacy": self.legacy_dir / rel_path,
                        "target": self.target_dir / rel_path,
                    }
                )
                logger.debug("Paired project: %s", rel_path)
            else:
                logger.warning(
                    "Legacy project %s has no target counterpart – skipped.",
                    rel_path,
                )

        logger.info(
            "Discovered %d project pairs out of %d legacy projects.",
            len(pairs),
            len(legacy_projects),
        )
        return pairs

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _clone_branch(self, branch: str, dest: Path) -> None:
        """Shallow-clone *branch* of the demo repo into *dest*."""
        if dest.exists():
            logger.info("Clone already exists at %s – skipping.", dest)
            return

        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning branch '%s' into %s …", branch, dest)

        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            branch,
            self.repo_url,
            str(dest),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Clone complete: %s", dest)

    @staticmethod
    def _find_projects(root: Path) -> Dict[Path, Path]:
        """Return ``{relative_path: absolute_path}`` for every Godot project."""
        projects: Dict[Path, Path] = {}
        for godot_file in root.rglob("project.godot"):
            project_dir = godot_file.parent
            rel = project_dir.relative_to(root)
            projects[rel] = project_dir
        return projects
