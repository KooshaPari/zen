#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

# Parallel unit coverage via coverage.py + xdist (auto-detect plugin)
export COVERAGE_PROCESS_START=.coveragerc
workers="${PYTEST_WORKERS:-auto}"
dist="${PYTEST_DIST:-loadscope}"
expr="${PYTEST_UNIT_EXPRESSION:-not integration and not benchmark and not slow and not no_mock_provider}"

# Prefer the active interpreter to avoid PATH mismatches
PY_BIN="${PY_BIN:-python}"

# Load test overrides to avoid .env affecting tests
if [ -f .env.tests ]; then
  echo "[tests] Loading .env.tests overrides"
  set -a; . ./.env.tests; set +a
fi

# Honor port overrides if explicit URLs are not set
if [ -z "${NATS_SERVERS:-}" ] && [ -n "${NATS_PORT:-}" ]; then export NATS_SERVERS="nats://localhost:${NATS_PORT}"; fi
if [ -z "${REDIS_URL:-}" ] && [ -n "${REDIS_PORT:-}" ]; then export REDIS_URL="redis://localhost:${REDIS_PORT}/0"; fi

# Detect pytest-xdist availability; fall back to serial if absent
if "$PY_BIN" - <<'PY' 2>/dev/null
import importlib.util as u
import sys
sys.exit(0 if u.find_spec('xdist') else 1)
PY
then
  PARALLEL_ARGS=(-n "$workers" --dist "$dist")
else
  echo "[coverage] pytest-xdist not found; running in serial. (pip install -r requirements-dev.txt to enable parallel)"
  PARALLEL_ARGS=()
fi

echo "[coverage] Running unit tests with coverage ${PARALLEL_ARGS:+in parallel}..."
COVERAGE_FILE=.coverage.unit "$PY_BIN" -m coverage run -m pytest "${PARALLEL_ARGS[@]}" \
  -k "not integration" -m "$expr" \
  --ignore=simulator_tests/ --ignore=tests/integration

echo "[coverage] Combining unit worker coverage data..."
COVERAGE_FILE=.coverage.unit coverage combine

echo "[coverage] Generating unit coverage reports..."
COVERAGE_FILE=.coverage.unit "$PY_BIN" -m coverage report -m --fail-under=100
COVERAGE_FILE=.coverage.unit "$PY_BIN" -m coverage xml -o coverage-unit.xml
COVERAGE_FILE=.coverage.unit "$PY_BIN" -m coverage html -d htmlcov-unit

echo "[coverage] Unit coverage report: htmlcov-unit/index.html"
