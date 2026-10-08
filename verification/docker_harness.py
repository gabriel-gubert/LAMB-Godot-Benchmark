"""
Docker harness module for managing containerized Godot 4 execution environments.
"""

import os
import shutil
import subprocess
from typing import Optional, Dict, Any


class DockerHarness:
    """Manages container creation, workspace mounting, and command execution."""

    def __init__(self, image_name: str = "godot4-eval-harness:latest"):
        self.image_name = image_name
        self._check_docker_available()

    def _check_docker_available(self) -> None:
        """Verifies that the Docker CLI is installed and responsive."""
        try:
            subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as err:
            raise RuntimeError(
                "Docker is not available or the daemon is not running. "
                "Ensure Docker Desktop/Daemon is running."
            ) from err

    def run_in_container(
        self,
        project_dir: str,
        command: str,
        timeout: int = 60,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Runs a command inside a ephemeral Docker container with project_dir mounted to /workspace.

        Args:
            project_dir: Path to the target project directory on the host.
            command: Shell command string to execute inside the container.
            timeout: Maximum execution timeout in seconds.
            env_vars: Key-value dictionary of environment variables to pass into the container.

        Returns:
            subprocess.CompletedProcess containing stdout, stderr, and returncode.
        """
        abs_project_dir = os.path.abspath(project_dir)

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{abs_project_dir}:/workspace",
            "-w",
            "/workspace",
        ]

        if env_vars:
            for key, val in env_vars.items():
                docker_cmd.extend(["-e", f"{key}={val}"])

        docker_cmd.extend([self.image_name, "/bin/bash", "-c", command])

        try:
            result = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return result
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                args=docker_cmd,
                returncode=-1,
                stdout=exc.stdout or "",
                stderr="Container execution timed out.",
            )

    def prepare_migrated_project(self, target_project_dir: str, migrated_project_dir: str) -> None:
        """Copies the target project baseline structure into the persistent output directory if it does not exist."""
        if not os.path.exists(migrated_project_dir):
            shutil.copytree(target_project_dir, migrated_project_dir)

    def prepare_workspace(self, target_project_dir: str, script_relative_path: str, migrated_code: str) -> str:
        """Creates a temporary workspace copy of target_project_dir and overwrites the script file.

        Args:
            target_project_dir: Ground truth target project folder.
            script_relative_path: Relative path to the target script file (e.g. 'scripts/player.gd').
            migrated_code: Content of the generated/migrated GDScript file.

        Returns:
            Path to the temporary workspace directory.
        """
        temp_dir = f"{target_project_dir}_eval_tmp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        shutil.copytree(target_project_dir, temp_dir)

        target_script_path = os.path.join(temp_dir, script_relative_path)
        os.makedirs(os.path.dirname(target_script_path), exist_ok=True)

        with open(target_script_path, "w", encoding="utf-8") as f:
            f.write(migrated_code)

        return temp_dir

    def cleanup_workspace(self, workspace_dir: str) -> None:
        """Safely removes temporary workspace directories."""
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir, ignore_errors=True)