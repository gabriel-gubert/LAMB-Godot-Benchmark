#!/usr/bin/env bash
set -e

echo "======================================================="
echo " Godot Migration Evaluation Framework - Launch Pipeline"
echo "======================================================="

# 1. Run Migration Paradigms
echo "[1/3] Executing code migration pipelines..."
python3 main.py --run-baselines --paradigm all

# 2. Build Docker Container
echo "[2/3] Building Godot 4 evaluation Docker image..."
docker build -t godot4-eval-harness:latest .

# 3. Evaluate Containerized Harness
echo "[3/3] Running static analysis & dynamic runtime verification..."
python3 main.py --evaluate

echo "======================================================="
echo " Complete! Results saved to experiment_results.json"
echo "======================================================="
