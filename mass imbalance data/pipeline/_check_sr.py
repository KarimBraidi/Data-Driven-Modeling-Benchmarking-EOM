import json
with open('symbolic.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells:', len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    if 'pysr_sweep_results' not in src: continue
    ec = cell.get('execution_count')
    has_output = len(cell.get('outputs', [])) > 0
    first = src.split('\n')[0][:80]
    print(f'  Cell {i+1}: exec_count={ec}, has_output={has_output}')
    print(f'    First line: {first}')
