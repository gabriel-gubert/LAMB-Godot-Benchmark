"""Global configuration constants for the Godot Migration Benchmark."""

import os
from pathlib import Path

# ──────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
DATASET_DIR = PROJECT_ROOT / "dataset" / "repos"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"

# ──────────────────────────────────────────────────
# Git Repository Configuration
# ──────────────────────────────────────────────────
DEMO_REPO_URL = "https://github.com/godotengine/godot-demo-projects.git"
LEGACY_BRANCH = "3.x" # Godot 3.x branch
TARGET_BRANCH = "master" # Godot 4.x branch (master tracks latest stable)

# ──────────────────────────────────────────────────
# Docker Configuration
# ──────────────────────────────────────────────────
DOCKER_IMAGE = "godot4-eval-harness:latest"
DOCKER_WORKSPACE = "/workspace"

# ──────────────────────────────────────────────────
# Evaluation Parameters
# ──────────────────────────────────────────────────
REQUIRE_REVIEW_EVALUATION = True
DYNAMIC_QUIT_FRAMES = 120
USE_NATIVE_GDSCRIPT_CODEBLEU = True # Set True for native GDScript parser, False for standard Python mapping
CODEBLEU_LANG = "python" # Used when USE_NATIVE_GDSCRIPT_CODEBLEU is False
CODEBLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

# ──────────────────────────────────────────────────
# Hybrid RAG Parameters
# ──────────────────────────────────────────────────
RRF_K = 60 # Reciprocal Rank Fusion constant
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 4 # Number of context chunks after re-ranking

# ──────────────────────────────────────────────────
# LAMB CLI Parameters
# ──────────────────────────────────────────────────
LAMB_MAX_ATTEMPTS = 3
LAMB_MAPPING_ALIAS = "Godot-3-to-4"
LAMB_SUMMARIZATION_THRESHOLD = None
LAMB_VERBOSE = True
LAMB_CONFIG_PATH = OUTPUT_DIR / "LAMB_CONFIG.txt"

# ──────────────────────────────────────────────────
# Ensure directories exist
# ──────────────────────────────────────────────────
for d in [DATASET_DIR, OUTPUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
