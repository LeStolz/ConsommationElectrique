import json
import itertools

with open('notebooks/02_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'evaluator.grid_search_rolling_validation(SARIMAXRegressor' in src:
            new_src = '''import itertools

p = [0, 1, 2]
d = [0, 1]
q = [0, 1, 2]
orders = list(itertools.product(p, d, q))

P = [0, 1]
D = [0, 1]
Q = [0, 1, 2]
seasonal_orders = [(*x, 24) for x in itertools.product(P, D, Q)]

evaluator.grid_search_rolling_validation(SARIMAXRegressor, [
	{
		"order": orders,
		"seasonal_order": seasonal_orders,
		"features_cols": features_cols,
		"since_year": [2023]
	}
]);'''
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            # Remove the trailing newline from the last line
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
            break

with open('notebooks/02_evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print('Updated notebook!')
