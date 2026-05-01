#!/usr/bin/env python3
import json

# Read the notebook
with open('HoopSINDy_EOM_Discovery.ipynb', 'r') as f:
    nb = json.load(f)

# Get current first cell type
print(f"Before: First cell type = {nb['cells'][0].get('cell_type', 'unknown')}")

# Reverse the cell order (flip from 25 to 1 back to 1 to 25)
cells = nb['cells']
cells.reverse()

# Write back
with open('HoopSINDy_EOM_Discovery.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"After: First cell type = {nb['cells'][0].get('cell_type', 'unknown')}")
print(f"✓ Notebook cells reordered. Total cells: {len(cells)}")
