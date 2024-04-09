#!/bin/bash

cd ..
python evaluator.py csv  --log_dir ./mlruns/ --experiments "['645251578826055288', '287494210064575407', '801969899048079208']" --results_file results.csv