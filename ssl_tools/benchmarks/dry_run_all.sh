#!/bin/bash

# Run all benchmarks with dry-run mode enabled

# Benchmark A1
echo "************ Benchmark A1 ************"
./main_supervised.py --config ./configs/benchmarks/dry_run/benchmark_a1_dry_run.yaml

# Benchmark A2
echo "************ Benchmark A2 ************"
./main_supervised.py --config ./configs/benchmarks/dry_run/benchmark_a2_dry_run.yaml

echo "Finished running all benchmarks in dry-run mode"