#!/usr/bin/env bash
# Run MVP pipeline tests on the server.
#
# Usage:
#   ./scripts/run_tests.sh              # fast unit tests only (default)
#   ./scripts/run_tests.sh --all        # unit + integration + slow
#   ./scripts/run_tests.sh --integration # unit + integration (no model load)
#   ./scripts/run_tests.sh --slow       # only model tests
#   ./scripts/run_tests.sh --file test_guard_node.py  # single file

set -euo pipefail

cd /opt/projects/chemist-agent
source venv/bin/activate

PYTEST="python -m pytest"
ARGS="-v --tb=short"

case "${1:-}" in
  --all)
    echo "=== Running ALL tests (unit + integration + slow) ==="
    $PYTEST $ARGS -m "" "$@"
    ;;
  --integration)
    echo "=== Running unit + integration tests ==="
    $PYTEST $ARGS -m "not slow and not llm" "$@"
    ;;
  --slow)
    echo "=== Running slow (model) tests only ==="
    $PYTEST $ARGS -m "slow" "$@"
    ;;
  --file)
    shift
    FILE="real_proj/mvp/tests/${1}"
    echo "=== Running: $FILE ==="
    $PYTEST $ARGS -m "not slow and not llm" "$FILE"
    ;;
  "")
    echo "=== Running fast unit tests (no integration, no model) ==="
    $PYTEST $ARGS -m "not slow and not integration and not llm"
    ;;
  *)
    echo "Usage: $0 [--all|--integration|--slow|--file <filename>]"
    exit 1
    ;;
esac
