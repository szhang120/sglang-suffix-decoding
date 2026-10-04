#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install -r configs/cpu-requirements.txt
"$PYTHON_BIN" -m cmake -S native -B native/build \
  -DPython_EXECUTABLE="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/native/suffix_native"
"$PYTHON_BIN" -m cmake --build native/build -j 4
PYTHONPATH="$PWD/native" "$PYTHON_BIN" -m unittest discover -s tests -v
