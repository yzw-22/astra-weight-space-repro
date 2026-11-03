#!/usr/bin/env bash
# Full pipeline: sanity gate -> 3 games x 3 attempts -> results table.
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.mplcache"
PY=${PY:-python}

$PY check_reference.py --episodes 100
$PY run_experiment.py --game all --attempts 3 --episodes 1000
$PY report.py --runs-root runs --out ../docs/results.md
