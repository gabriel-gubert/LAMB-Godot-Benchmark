# Godot 3.x $\rightarrow$ 4.x Code Migration Benchmark

An automated, containerized pipeline to evaluate GDScript code migration performance across major engine version updates. This repository benchmarks zero-shot LLMs, Advanced Hybrid RAG architectures, and the **LAMB** framework against official Godot demonstration repositories under strict scene-tree structural constraints.

## Architecture Overview

```
                               ┌───────────────────────────┐
                               │ Godot Demo Repos (GitHub) │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │    Isomorphism Auditor    │
                               │  (Case & Topology Check)  │
                               └─────────────┬─────────────┘
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   │ Passed Projects                                   │ Review Flagged
                   ▼                                                   ▼
      ┌─────────────────────────┐                         ┌─────────────────────────┐
      │ dataset/audited_passed/ │                         │ require_manual_review/  │
      │ (Legacy & Target copies)│                         │ (1:1 Node Type Shifts)  │
      └────────────┬────────────┘                         └─────────────────────────┘
                   │ Valid Pairs
                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               Migration Baseline Execution                              │
│             Copies ground-truth project assets -> output//<project_name>/               │
└───────────────┬────────────────────────────┬────────────────────────────┬───────────────┘
                │                            │                            │
                ▼                            ▼                            ▼
         ┌──────────────┐          ┌───────────────────┐          ┌──────────────┐
         │ Baseline 1   │          │ Baseline 2        │          │ Baseline 3   │
         │ Zero-Shot    │          │ Hybrid RAG        │          │ LAMB         │
         │ Generation   │          │ (AST+HyDE+RRF+CE) │          │ Framework    │
         └──────┬───────┘          └─────────┬─────────┘          └───────┬──────┘
                │                            │                            │
                └────────────────────────────┼────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Containerized Harness   │
                               │   (Godot 4 Headless CLI)  │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
            ┌─────────────────────┐                     ┌────────────────────┐
            │   Static Analysis   │                     │  Dynamic Runtime   │
            │  - Syntax Check     │                     │  - Anti-crash      │
            │  - AST Edit Dist    │                     │  - 120 Frames Run  │
            │  - CodeBLEU         │                     │  - Pass/Fail Rate  │
            └──────────┬──────────┘                     └──────────┬─────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Visual Summaries & LaTeX  │
                               │ (JSON, Rich Tables, .tex) │
                               └───────────────────────────┘
```

## Key Features

- **Automated Dataset Curation**: Clones official `godot-demo-projects` branches (`3.x` legacy and `master` target).
- **Flexible Isomorphism Auditing**: Enforces structural equivalence with case-insensitive file matching, order-invariant tree topology matching, and node-name invariance.
- **Isolated Review Routing**: Moves projects with $1:1$ node type changes into `dataset/require_manual_review/` and writes dedicated `audit_report.json` files directly into project folders for manual inspection.
- **Full Project Persistence**: Persists full, playable Godot 4 demo projects to `output/<paradigm>/<project_name>/` so you can open and run them in the Godot 4 editor.
- **Advanced Hybrid RAG**: Implements AST symbol extraction, Hypothetical Document Embeddings (HyDE), combined dense FAISS + sparse BM25 search fused with Reciprocal Rank Fusion (RRF), and Cross-Encoder re-ranking.
- **LAMB Integration & Pre-Configuration**: Ships with pre-configured `.lambconfig.toml` and relational mapping table files for reproducible migration runs.
- **Containerized Verification**: Runs generated GDScript in isolated Docker containers using the native Godot 4 headless engine CLI.
- **Comprehensive Output Suite**: Generates Rich console tables, detailed JSON reports (`experiment_results.json`), and publication-ready LaTeX tables (`summary_table.tex`, `per_demo_table.tex`).

## Prerequisites

Before running the benchmark, ensure you have the following installed:

1. **Docker Desktop / Docker Engine** (running and responsive)
2. **Python 3.10+**
3. **Git**
4. An **OpenAI API Key** (configured in environment variables or `instances.py`)

## Installation & Configuration

1. **Clone this repository:**
```bash
git clone [https://github.com/gabriel-gubert/LAMB-Godot-Benchmark.git](https://github.com/gabriel-gubert/LAMB-Godot-Benchmark.git)
cd LAMB-Godot-Benchmark

```

2. **Clone and install LAMB from the project root:**

```bash
git clone [https://www.github.com/gabriel-gubert/LAMB.git](https://www.github.com/gabriel-gubert/LAMB.git)
pip install -e ./LAMB

```

3. **Install Benchmark dependencies:**

```bash
pip install -r requirements.txt

```

4. **Verify Configuration Files:**
Ensure the following pre-configured shipping files are located in the project root:

- `.lambconfig.toml` — LAMB global configuration file.
- Godot Mapping Table (e.g., `godot_mapping_table.json`).

5. **Configure API Keys:**
Set your OpenAI API key in your environment:

```bash
export OPENAI_API_KEY="your-api-key-here"

```

## Quick Start

### Option 1: Automated Execution Script (Recommended)

Run the automated launcher script, which clones LAMB, builds the Docker image, audits dataset isomorphism, executes migration paradigms, and exports results:

- **Linux / macOS:**

```bash
chmod +x run_experiments.sh
./run_experiments.sh

```

- **Windows:**

```cmd
run_experiments.bat

```

### Option 2: Step-by-Step CLI Execution (`main.py`)

#### 1. Build the Godot 4 Evaluation Container

```bash
docker build -t godot4-eval-harness:latest .

```

#### 2. Download Repositories & Run Isomorphism Audit

```bash
python main.py --audit-dataset

```

Passed projects are copied to `dataset/audited_passed/`, while flagged projects are moved to `dataset/require_manual_review/`.

#### 3. Execute Migration Baselines

Run all paradigms (including `lamb` using `.lambconfig.toml`):

```bash
python main.py --run-baselines --paradigm all

```

This builds full, playable Godot 4 projects under `output/<paradigm>/<project_name>/`.

#### 4. Run Static Analysis & Dynamic Runtime Verification

```bash
python main.py --evaluate --report-output experiment_results.json

```

## Directory Structure

```
.
├── .lambconfig.toml                  # Shipped LAMB configuration file
├── godot_mapping.json                # Shipped Godot 3.x -> 4.x relational mapping table
├── LAMB/                             # Cloned LAMB framework repository
├── dataset/                          # Curation & Workspace Routing
│   ├── audited_passed/               # Copied 1:1 passed legacy and target demo projects
│   ├── require_manual_review/        # Projects flagged for manual review (1:1 node type shifts)
│   ├── downloader.py                 # Demo repository downloader
│   └── isomorphism_auditor.py        # Case-insensitive topology auditor
├── output/                           # Full Persisted Godot 4 Projects
│   ├── zero-shot/                    # Playable Godot 4 projects generated by Zero-Shot
│   ├── hybrid-rag/                   # Playable Godot 4 projects generated by Hybrid RAG
│   └── lamb/                         # Playable Godot 4 projects generated by LAMB
├── baselines/                        # Migration Paradigm Implementations
│   ├── zero_shot.py                  # Baseline 1: Zero-Shot LLM Generation
│   ├── hybrid_rag.py                 # Baseline 2: AST + HyDE + FAISS/BM25 RRF + Cross-Encoder
│   └── lamb_runner.py                # Proposed LAMB CLI Integration (`lamb migrate`)
├── verification/                     # Containerized Evaluation Harness
│   ├── docker_harness.py             # Docker container lifecycle management
│   ├── static_analyzer.py            # Godot check-only, AST edit distance, CodeBLEU
│   ├── gdscript_codebleu.py          # Native Tree-Sitter GDScript CodeBLEU calculation
│   └── dynamic_runner.py             # 120-frame headless anti-crash runtime execution
├── config.py                         # Global benchmark constants & hyper-parameters
├── instances.py                      # LangChain LLM & Embedding configurations
├── main.py                           # Benchmark orchestrator CLI
├── Dockerfile                        # Ubuntu 22.04 + Godot 4 Headless CLI environment
├── requirements.txt                  # Python dependencies
├── run_experiments.sh                # Shell launcher for Linux/macOS
└── run_experiments.bat               # Batch launcher for Windows

```

## License

Distributed under the MIT License. See `LICENSE` for details.