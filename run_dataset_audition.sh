#!/usr/bin/env bash
set -e

echo "======================================================="
echo " Godot Migration Evaluation Framework - Audit Pipeline "
echo "======================================================="

echo "Downloading demo projects and executing isomorphism audit..."
python3 main.py --audit-dataset