"""
Dynamic runner module for evaluating operational stability and runtime anti-crash behavior.
"""

import time
from typing import Dict, Any
from verification.docker_harness import DockerHarness


class DynamicRunner:
    """Evaluates dynamic runtime execution of migrated projects inside headless Godot 4 containers."""

    def __init__(self, docker_harness: DockerHarness, frame_limit: int = 120):
        self.harness = docker_harness
        self.frame_limit = frame_limit

    def evaluate_runtime(
        self,
        migrated_project_dir: str,
        timeout: int = 45,
    ) -> Dict[str, Any]:
        """Runs the persisted project headlessly for `frame_limit` frames inside container."""
        cmd = (
            f"godot4 --headless --audio-driver Dummy --display-driver headless "
            f"--quit-after {self.frame_limit}"
        )

        start_time = time.time()
        process_res = self.harness.run_in_container(
            project_dir=migrated_project_dir,
            command=cmd,
            timeout=timeout,
        )
        latency = time.time() - start_time

        passed = False
        if process_res.returncode == 0 and "SCRIPT ERROR" not in process_res.stderr and "CRASH" not in process_res.stderr:
            passed = True

        return {
            "passed": passed,
            "returncode": process_res.returncode,
            "latency_seconds": round(latency, 4),
            "stdout": process_res.stdout,
            "stderr": process_res.stderr,
        }