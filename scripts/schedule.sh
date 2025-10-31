#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/main.py --multirun trainer.SYNQ=0.999,0.975,0.95,0.925,0.9,0.8,0.7 optimization.gamma=0.1,0.25,0.5,0.75,0.9

