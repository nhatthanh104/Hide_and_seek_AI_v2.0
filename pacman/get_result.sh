#!/usr/bin/env bash
# Run ONE game and return the winner

SEEKER="${1:-example_student}"
HIDER="${2:-example_student}"
NO_VIZ="${NO_VIZ:-true}"
SUBMISSIONS_DIR="${SUBMISSIONS_DIR:-submissions}"
STEP_TIMEOUT="${STEP_TIMEOUT:-3.0}"
CAPTURE_DISTANCE="${CAPTURE_DISTANCE:-2}"
PACMAN_SPEED="${PACMAN_SPEED:-2}"



# Auto-detect Python command
# Try to find Python in conda env or system Python
PYTHON_CMD=""

# Try conda env python first (common location)
for python_path in \
    "/c/Users/$USER/miniconda3/envs/ml/python.exe" \
    "/c/Users/$USER/anaconda3/envs/ml/python.exe" \
    "$HOME/miniconda3/envs/ml/python.exe" \
    "$HOME/anaconda3/envs/ml/python.exe"; do
    if [ -f "$python_path" ]; then
        PYTHON_CMD="$python_path"
        break
    fi
done

# Fallback to system python
if [ -z "$PYTHON_CMD" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python not found!" >&2
        exit 1
    fi
fi

# Run game with UTF-8 encoding
export PYTHONIOENCODING=utf-8
output="$(cd src && $PYTHON_CMD arena.py \
    --seek "$SEEKER" \
    --hide "$HIDER" \
    --submissions-dir "../$SUBMISSIONS_DIR" \
    --step-timeout "$STEP_TIMEOUT" \
    --capture-distance "$CAPTURE_DISTANCE" \
    --pacman-speed "$PACMAN_SPEED" \
    $( [ "$NO_VIZ" = true ] && echo --no-viz ) 2>&1)"

# Parse result - check các pattern trong arena.py's display_results()
# Extract winner and total steps

# Determine winner
winner=""
if echo "$output" | grep -q "(Pacman)"; then
    winner="pacman_wins"
elif echo "$output" | grep -q "(Ghost)"; then
    winner="ghost_wins"
elif echo "$output" | grep -qi "draw"; then
    winner="draw"
else
    winner="error"
fi

# Extract total steps from output
# Pattern: "  Total Steps: 42"
total_steps=$(echo "$output" | grep -i "Total Steps:" | sed -E 's/.*Total Steps:\s*([0-9]+).*/\1/')

# If steps not found, set to -1
if [ -z "$total_steps" ]; then
    total_steps="-1"
fi

# Output format: winner:steps
echo "${winner}:${total_steps}"

# Debug output if error
if [ "$winner" = "error" ]; then
    echo "=== DEBUG: Could not parse winner from output ===" >&2
    echo "$output" >&2
    echo "=== END DEBUG ===" >&2
fi