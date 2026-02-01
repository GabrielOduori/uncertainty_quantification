#!/bin/bash
# Reproduce Paper Results - Convenience Script
#
# This script runs all experiments to reproduce dissertation chapter results.

set -e  # Exit on error

echo "================================================================================"
echo "  REPRODUCING PAPER RESULTS: Rigorous UQ for Air Quality Fusion"
echo "================================================================================"
echo ""

# Check if scipy is installed
python -c "import scipy" 2>/dev/null || {
    echo "⚠️  Warning: scipy not installed. Installing dependencies..."
    pip install scipy pandas numpy
    echo ""
}

# Run experiments
echo "Running experiments..."
echo ""

python experiments/reproduce_paper.py

echo ""
echo "================================================================================"
echo "✅ COMPLETE! Results saved to results/"
echo "================================================================================"
echo ""
echo "View results:"
echo "  cat results/paper_results.txt"
echo ""
echo "View decision report:"
echo "  cat results/tables/decision_report.csv"
echo ""
