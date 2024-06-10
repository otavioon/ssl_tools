#!/bin/bash

echo "Running benchmark A1"
./main_supervised.py --config configs/benchmarks/benchmark_a1.yaml

echo "Analyzing results"
./main_supervised_analysis.py --config configs/benchmarks/analysis/benchmark_a1.yaml

echo "Done"
echo "Results are in results/"
