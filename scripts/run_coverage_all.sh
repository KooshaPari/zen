#!/usr/bin/env bash
set -euo pipefail

# Full coverage across unit + integration, combining results via coverage.py
export COVERAGE_PROCESS_START=.coveragerc

workers_unit="${PYTEST_WORKERS:-auto}"
dist_unit="${PYTEST_DIST:-loadscope}"
workers_e2e="${PYTEST_WORKERS_INTEGRATION:-2}"
dist_e2e="${PYTEST_DIST_INTEGRATION:-loadscope}"
expr_unit="${PYTEST_UNIT_EXPRESSION:-not integration and not benchmark and not slow and not no_mock_provider}"

PY_BIN="${PY_BIN:-python}"

# Load test overrides to avoid .env affecting tests
if [ -f .env.tests ]; then
  echo "[tests] Loading .env.tests overrides"
  set -a; . ./.env.tests; set +a
fi

# Honor port overrides if explicit URLs are not set
if [ -z "${NATS_SERVERS:-}" ] && [ -n "${NATS_PORT:-}" ]; then export NATS_SERVERS="nats://localhost:${NATS_PORT}"; fi
if [ -z "${REDIS_URL:-}" ] && [ -n "${REDIS_PORT:-}" ]; then export REDIS_URL="redis://localhost:${REDIS_PORT}/0"; fi

echo "[coverage] Cleaning previous coverage artifacts..."
rm -rf .coverage* htmlcov htmlcov-* coverage*.xml || true

# Detect pytest-xdist availability; set parallel args per suite
if "$PY_BIN" - <<'PY' 2>/dev/null
import importlib.util as u
import sys
sys.exit(0 if u.find_spec('xdist') else 1)
PY
then
  PARALLEL_UNIT=(-n "$workers_unit" --dist "$dist_unit")
  PARALLEL_E2E=(-n "$workers_e2e" --dist "$dist_e2e")
else
  echo "[coverage] pytest-xdist not found; running serially."
  PARALLEL_UNIT=()
  PARALLEL_E2E=()
fi

echo "[coverage] Running unit tests ${PARALLEL_UNIT:+in parallel}..."
COVERAGE_FILE=.coverage.unit "$PY_BIN" -m coverage run -m pytest "${PARALLEL_UNIT[@]}" \
  -k "not integration" -m "$expr_unit" --ignore=simulator_tests/ --ignore=tests/integration

echo "[coverage] Combining unit worker coverage data..."
COVERAGE_FILE=.coverage.unit "$PY_BIN" -m coverage combine

echo "[coverage] Running integration/E2E tests ${PARALLEL_E2E:+in parallel}..."
set +e
COVERAGE_FILE=.coverage.e2e "$PY_BIN" -m coverage run -m pytest "${PARALLEL_E2E[@]}" tests/integration
set -e

echo "[coverage] Combining E2E worker coverage data..."
COVERAGE_FILE=.coverage.e2e "$PY_BIN" -m coverage combine

echo "[coverage] Combining coverage data and generating reports..."
"$PY_BIN" -m coverage combine .coverage.unit .coverage.e2e
"$PY_BIN" -m coverage report -m --fail-under=100
"$PY_BIN" -m coverage xml -o coverage-combined.xml
"$PY_BIN" -m coverage html -d htmlcov

echo "[coverage] Combined coverage report: htmlcov/index.html"
