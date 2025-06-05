#!/bin/bash

# Output log file
LOG_FILE="batch_eval_output.txt"
: > "$LOG_FILE"  # Truncate the log file if it already exists

# Directory containing all *_gray folders
EVAL_ROOT="../davis2017eval"

# Path to DAVIS ground truth
DAVIS_PATH="../FrameSkipSAM/data/DAVIS"

# Loop over all *_gray folders
for RESULTS_DIR in "$EVAL_ROOT"/sam2_preds_*_gray; do
    if [ -d "$RESULTS_DIR" ]; then
        echo "Evaluating: $RESULTS_DIR" | tee -a "$LOG_FILE"
        {
            echo "========================"
            echo "Evaluating: $RESULTS_DIR"
            python ../davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path "$RESULTS_DIR" --davis_path "$DAVIS_PATH"
            echo "Done with: $RESULTS_DIR"
            echo
        } >> "$LOG_FILE" 2>&1
    fi
done

echo "All evaluations completed. See log in $LOG_FILE"
