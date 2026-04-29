import json
with open('pipeline/SINDy_Pipeline.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('id') == '#VSC-7a8300c3':
        for o in cell.get('outputs', []):
            lines = ''.join(o.get('text', [])).splitlines()
            for l in lines:
                print(l)
        break
