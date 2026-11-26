#!/usr/bin/env bash
cd "$(dirname "$0")"
for p in python3.11 python3.10 python3.9 python3 python; do
  if command -v "$p" >/dev/null 2>&1; then exec "$p" start.py "$@"; fi
done
echo "Python 3.9-3.11 not found. Install Python 3.11 from python.org."
