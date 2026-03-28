#!/usr/bin/env bash
# Scripted killer-demo session for asciinema recording.
# Run from repo root:
#   asciinema rec assets/demo.cast --command "bash assets/record_demo.sh" --overwrite
set -e
cd "$(dirname "$0")/.."

# Brief pause helper
p() { sleep "$1"; }

clear
echo "🧠 Embedding Debugger — killer demo"
echo "===================================="; p 1.2

echo ""
echo "$ python -m demo.killer_demo --model all-MiniLM-L6-v2"; p 0.8
python -m demo.killer_demo --model all-MiniLM-L6-v2 --no-export 2>/dev/null
p 2
