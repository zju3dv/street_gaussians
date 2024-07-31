#!/bin/bash
scenes=("006"  "026" "090" "105" "108" "134" "150" "181") 
for scene in "${scenes[@]}"; do
    python render.py --config configs/experiments_waymo/waymo_val_$scene.yaml mode evaluate
    python render.py --config configs/experiments_waymo/waymo_val_$scene.yaml mode trajectory
done
