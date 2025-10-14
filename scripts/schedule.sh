#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/main.py trainer.SYNQ=0.7,0.8,0.9,0.925,0.95,0.975,0.999

