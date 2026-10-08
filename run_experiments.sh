#!/usr/bin/env bash
set -e

echo "======================================================="
echo " Godot Migration Evaluation Framework - Launch Pipeline"
echo "======================================================="

# 1. Check & Install Dependencies
echo "[1/5] Installing Python requirements..."
python3 -m pip install -r requirements.txt --quiet

# 2. Build Docker Container
echo "[2/5] Building Godot 4 evaluation Docker image..."
docker build -t godot4-eval-harness:latest .

# 3. Audit Dataset
echo "[3/5] Downloading demo projects and executing isomorphism audit..."
python3 main.py --audit-dataset

# 4. Run Migration Paradigms
echo "[4/5] Executing code migration pipelines..."
python3 main.py --run-baselines --paradigm zero-shot
python3 main.py --run-baselines --paradigm hybrid-rag
python3 main.py --run-baselines --paradigm lamb

# 5. Evaluate Containerized Harness
echo "[5/5] Running static analysis & dynamic runtime verification..."
python3 main.py --evaluate --report-output ./experiment_results.json

echo "======================================================="
echo " Complete! Results saved to experiment_results.json"
echo "======================================================="
