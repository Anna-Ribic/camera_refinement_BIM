#!/bin/bash
directory="../../ConSLAM/data/lidar/lidar-depth-dense-2/lidar-depth-dense-2"

for file in "$directory"/*; do
  if [ -f "$file" ]; then
    python runnet.py --cur "$file"
  fi
done
