"""Baseline 1: Zero-Shot Parametric Generation.

Passes full legacy GDScript source to the chat model with a zero-shot prompt,
relying strictly on the model's internal parametric knowledge of the Godot
3.x → 4.x migration.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from instances import chat_model

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert GDScript developer specializing in \
migrating Godot Engine projects from version 3.x to version 4.x.

Your task is to convert the provided Godot 3.x GDScript file into valid \
Godot 4.x GDScript.  Apply ALL necessary API changes, signal syntax updates, \
type renames, property renames, and method signature changes based on your \
knowledge of the Godot 4 migration.

Return ONLY the converted GDScript code.  Do NOT include explanations, \
commentary, markdown fences, or any text that is not valid GDScript."""

_USER_TEMPLATE = """\
Convert the following Godot 3.x GDScript file to Godot 4.x.

--- BEGIN Godot 3.x GDScript ---
{source_code}
--- END Godot 3.x GDScript ---

Return ONLY the converted Godot 4.x GDScript code."""


class ZeroShotGenerator:
    """Generates Godot 4 code from legacy source using zero-shot prompting."""

    def __init__(self, model=None):
        self.model = model or chat_model

    def migrate_file(
        self,
        legacy_path: Path,
        output_path: Optional[Path] = None,
    ) -> Dict:
        """Migrate a single GDScript file.

        Parameters
        ----------
        legacy_path : Path
            Path to the Godot 3.x ``.gd`` file.
        output_path : Path, optional
            Where to write the migrated code.  If *None* the result is
            returned but not persisted.

        Returns
        -------
        dict
            ``{"generated_code": str, "latency": float, "tokens": dict}``
        """
        source_code = legacy_path.read_text(errors="replace")
        user_content = _USER_TEMPLATE.format(source_code=source_code)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        logger.info("Zero-shot migrating: %s", legacy_path.name)
        t0 = time.perf_counter()
        response = self.model.invoke(messages)
        latency = time.perf_counter() - t0

        generated_code = response.content.strip()

        # Strip markdown fences if the model wraps its output
        generated_code = self._strip_code_fences(generated_code)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(generated_code)
            logger.info("Written migrated file to %s", output_path)

        token_usage = {}
        if hasattr(response, "response_metadata"):
            token_usage = response.response_metadata.get("token_usage", {})

        return {
            "generated_code": generated_code,
            "latency": latency,
            "tokens": token_usage,
        }

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences (```gdscript ... ```)."""
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
