#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/main.py experiment.trainer.SYNQ=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

