#!/usr/bin/env bash
# Run multiple games and collect statistics
# Usage: ./run_batch_games.sh [num_runs] [seeker_id] [hider_id]

# Configuration
RUNS="${1:-100}"                                    # Default 100 games
SEEKER="${2:-example_student}"                      # Pacman agent
HIDER="${3:-example_student}"                       # Ghost agent
LOGFILE="${LOGFILE:-log/batch_run_$(date +%Y%m%d_%H%M%S).log}"

# Counters
pacman_wins=0
ghost_wins=0
draws=0
errors=0

# Initialize logfile
echo "Batch run started: $(date)" > "$LOGFILE"
echo "Matchup: Pacman='$SEEKER' vs Ghost='$HIDER'" >> "$LOGFILE"
echo "Total runs: $RUNS" >> "$LOGFILE"
echo "----------------------------------------" >> "$LOGFILE"
echo "" >> "$LOGFILE"

echo "Running $RUNS games: Pacman='$SEEKER' vs Ghost='$HIDER'"
echo "Log file: $LOGFILE"
echo ""

# Run games
for i in $(seq 1 "$RUNS"); do
    # Show progress every 10 games
    if [ $((i % 10)) -eq 0 ] || [ $i -eq 1 ]; then
        echo "Progress: $i/$RUNS games completed..."
    fi
    
    # Run one game and capture result
    result=$(./get_result.sh "$SEEKER" "$HIDER" 2>&1)
    
    # Log the result
    echo "Game $i: $result" >> "$LOGFILE"
    
    # Count results
    case "$result" in
        pacman_wins)
            pacman_wins=$((pacman_wins+1))
            ;;
        ghost_wins)
            ghost_wins=$((ghost_wins+1))
            ;;
        draw)
            draws=$((draws+1))
            ;;
        *)
            errors=$((errors+1))
            echo "  [WARNING] Game $i returned unexpected result: $result" >> "$LOGFILE"
            ;;
    esac
done

echo ""
echo "All games completed!"
echo ""

# Calculate percentages
pacman_pct=$(awk "BEGIN {printf \"%.1f\", ($pacman_wins/$RUNS)*100}")
ghost_pct=$(awk "BEGIN {printf \"%.1f\", ($ghost_wins/$RUNS)*100}")
draw_pct=$(awk "BEGIN {printf \"%.1f\", ($draws/$RUNS)*100}")

# Print summary to console
echo "========================================="
echo "         STATISTICS SUMMARY"
echo "========================================="
echo "Total games    : $RUNS"
echo "Pacman wins    : $pacman_wins ($pacman_pct%)"
echo "Ghost wins     : $ghost_wins ($ghost_pct%)"
echo "Draws          : $draws ($draw_pct%)"
echo "Errors         : $errors"
echo "========================================="
echo ""
echo "Detailed log saved to: $LOGFILE"

# Append summary to logfile
echo "" >> "$LOGFILE"
echo "----------------------------------------" >> "$LOGFILE"
echo "FINAL STATISTICS:" >> "$LOGFILE"
echo "Total games    : $RUNS" >> "$LOGFILE"
echo "Pacman wins    : $pacman_wins ($pacman_pct%)" >> "$LOGFILE"
echo "Ghost wins     : $ghost_wins ($ghost_pct%)" >> "$LOGFILE"
echo "Draws          : $draws ($draw_pct%)" >> "$LOGFILE"
echo "Errors         : $errors" >> "$LOGFILE"
echo "Batch run finished: $(date)" >> "$LOGFILE"
