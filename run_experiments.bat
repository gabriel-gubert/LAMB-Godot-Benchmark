@echo off
SETLOCAL EnableDelayedExpansion

echo =======================================================
echo  Godot Migration Evaluation Framework - Launch Pipeline
echo =======================================================

echo [1/5] Installing Python requirements...
python -m pip install -r requirements.txt --quiet

echo [2/5] Building Godot 4 evaluation Docker image...
docker build -t godot4-eval-harness:latest .

echo [3/5] Downloading demo projects and executing isomorphism audit...
python main.py --audit-dataset

echo [4/5] Executing code migration pipelines...
python main.py --run-baselines --paradigm zero-shot
python main.py --run-baselines --paradigm hybrid-rag
python main.py --run-baselines --paradigm lamb

echo [5/5] Running static analysis ^& dynamic runtime verification...
python main.py --evaluate --report-output .\experiment_results.json

echo =======================================================
echo  Complete! Results saved to experiment_results.json
echo =======================================================
ENDLOCAL
pause
