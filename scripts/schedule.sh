#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python babylm/train.py trainer.max_epochs=5 logger=csv

python babylm/train.py trainer.max_epochs=10 logger=csv
