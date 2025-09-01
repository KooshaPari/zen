#!/usr/bin/env bash
set -euo pipefail

# Parallel E2E/Integration coverage via coverage.py + xdist (auto-detect plugin)
export COVERAGE_PROCESS_START=.coveragerc
workers="${PYTEST_WORKERS_INTEGRATION:-2}"
dist="${PYTEST_DIST_INTEGRATION:-loadscope}"

PY_BIN="${PY_BIN:-python}"

# Load test overrides to avoid .env affecting tests
if [ -f .env.tests ]; then
  echo "[tests] Loading .env.tests overrides"
  set -a; . ./.env.tests; set +a
fi

# Honor port overrides if explicit URLs are not set
if [ -z "${NATS_SERVERS:-}" ] && [ -n "${NATS_PORT:-}" ]; then export NATS_SERVERS="nats://localhost:${NATS_PORT}"; fi
if [ -z "${REDIS_URL:-}" ] && [ -n "${REDIS_PORT:-}" ]; then export REDIS_URL="redis://localhost:${REDIS_PORT}/0"; fi

if "$PY_BIN" - <<'PY' 2>/dev/null
import importlib.util as u
import sys
sys.exit(0 if u.find_spec('xdist') else 1)
PY
then
  PARALLEL_ARGS=(-n "$workers" --dist "$dist")
else
  echo "[coverage] pytest-xdist not found; running E2E serially."
  PARALLEL_ARGS=()
fi

echo "[coverage] Running E2E/integration tests ${PARALLEL_ARGS:+in parallel} with coverage..."
COVERAGE_FILE=.coverage.e2e "$PY_BIN" -m coverage run -m pytest "${PARALLEL_ARGS[@]}" tests/integration

echo "[coverage] Combining E2E worker coverage data..."
COVERAGE_FILE=.coverage.e2e "$PY_BIN" -m coverage combine

echo "[coverage] Generating E2E coverage reports..."
COVERAGE_FILE=.coverage.e2e "$PY_BIN" -m coverage report -m --fail-under=100
COVERAGE_FILE=.coverage.e2e "$PY_BIN" -m coverage xml -o coverage-e2e.xml
COVERAGE_FILE=.coverage.e2e "$PY_BIN" -m coverage html -d htmlcov-e2e

echo "[coverage] E2E coverage report: htmlcov-e2e/index.html"
