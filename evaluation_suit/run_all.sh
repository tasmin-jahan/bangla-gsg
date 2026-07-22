#!/bin/bash
set -e
echo "Installing evaluation dependencies..."
pip install -r evaluation_suit/requirements.txt
echo "Running the full evaluation suite..."
python -m evaluation_suit.eval.run_all
echo "Aggregating results..."
python -m evaluation_suit.scripts.aggregate_results
echo "Done!"
