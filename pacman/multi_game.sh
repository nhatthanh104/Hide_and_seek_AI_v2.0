#!/bin/bash

# Multi-game statistics script for Pacman vs Ghost Arena
# Usage: ./multi_game.sh <num_games> <seeker_id> <hider_id> [arena options]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

if [ "$#" -lt 3 ]; then
	echo -e "${RED}Usage: $0 <num_games> <seeker_id> <hider_id> [arena options]${NC}"
	echo ""
	echo "Examples:"
	echo "  $0 50 thanh_agent example_student"
	echo "  $0 100 thanh_agent example_student --capture-distance 2 --pacman-speed 2"
	exit 1
fi

NUM_GAMES="$1"
SEEKER="$2"
HIDER="$3"
shift 3

# Validate number of games
if ! [[ "$NUM_GAMES" =~ ^[0-9]+$ ]] || [ "$NUM_GAMES" -lt 1 ]; then
	echo -e "${RED}Error: Number of games must be a positive integer${NC}"
	exit 1
fi

# Statistics counters
PACMAN_WINS=0
GHOST_WINS=0
TIMEOUTS=0
ERRORS=0

# Get current directory
CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"

# Initialize conda for bash if available
CONDA_SCRIPT=""
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
	CONDA_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
	CONDA_SCRIPT="$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/c/Users/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
	CONDA_SCRIPT="/c/Users/$USER/miniconda3/etc/profile.d/conda.sh"
fi

if [ -n "$CONDA_SCRIPT" ]; then
	source "$CONDA_SCRIPT"
fi

# Determine Python command
if command -v conda >/dev/null 2>&1; then
	PYTHON_CMD="conda run -n ml python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
	PYTHON_CMD="python"
else
	echo -e "${RED}Error: Python not found${NC}"
	exit 1
fi

# Print header
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Running $NUM_GAMES games...${NC}"
echo -e "${CYAN}========================================${NC}"

cd "$SRC_DIR"

# Run games silently
START_TIME=$(date +%s)
for ((i=1; i<=$NUM_GAMES; i++)); do
	# Show progress
	echo -ne "Progress: $i/$NUM_GAMES\r"
	
	# Run game and capture output
	OUTPUT=$($PYTHON_CMD arena.py --seek "$SEEKER" --hide "$HIDER" --no-viz --delay 0 "$@" 2>&1 || true)
	
	# Parse result
	if echo "$OUTPUT" | grep -q "pacman_wins"; then
		((PACMAN_WINS++))
	elif echo "$OUTPUT" | grep -q "ghost_wins"; then
		((GHOST_WINS++))
	elif echo "$OUTPUT" | grep -q "timeout"; then
		((TIMEOUTS++))
	else
		((ERRORS++))
	fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Return to original directory
cd "$CURRENT_DIR"

# Print results
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  RESULTS${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "Seeker (Pacman): ${GREEN}$SEEKER${NC}"
echo -e "Hider (Ghost):   ${YELLOW}$HIDER${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "Total games:     ${BLUE}$NUM_GAMES${NC}"
echo -e "Execution time:  ${CYAN}${ELAPSED}s${NC} (avg: $(awk "BEGIN {printf \"%.2f\", $ELAPSED/$NUM_GAMES}")s/game)"
echo ""
echo -e "${GREEN}Pacman wins:     $PACMAN_WINS${NC} ($(awk "BEGIN {printf \"%.1f\", ($PACMAN_WINS/$NUM_GAMES)*100}")%)"
echo -e "${YELLOW}Ghost wins:      $GHOST_WINS${NC} ($(awk "BEGIN {printf \"%.1f\", ($GHOST_WINS/$NUM_GAMES)*100}")%)"

if [ $TIMEOUTS -gt 0 ] || [ $ERRORS -gt 0 ]; then
	echo -e "${RED}Timeouts:        $TIMEOUTS${NC} ($(awk "BEGIN {printf \"%.1f\", ($TIMEOUTS/$NUM_GAMES)*100}")%)"
	echo -e "${RED}Errors:          $ERRORS${NC} ($(awk "BEGIN {printf \"%.1f\", ($ERRORS/$NUM_GAMES)*100}")%)"
fi

echo -e "${CYAN}========================================${NC}"

# Determine winner
if [ $PACMAN_WINS -gt $GHOST_WINS ]; then
	echo -e "${GREEN}🏆 WINNER: PACMAN ($SEEKER)${NC}"
elif [ $GHOST_WINS -gt $PACMAN_WINS ]; then
	echo -e "${YELLOW}🏆 WINNER: GHOST ($HIDER)${NC}"
else
	echo -e "${BLUE}🤝 TIE GAME${NC}"
fi

echo -e "${CYAN}========================================${NC}"
