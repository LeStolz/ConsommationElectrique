import re

with open('src/models/baselines.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: if len(s_j) == 0:
replacement1 = '''            if len(s_j) == 0:
                final_prediction.loc[df_day.index] = df_day['consommation_mw_moins_7']
                continue'''
content = re.sub(
    r'            if len\(s_j\) == 0:\n                final_prediction\n                continue',
    replacement1,
    content
)

# Fix 2: after for d in candidate_dates loop, check if candidates is empty
replacement2 = '''
            if len(candidates) == 0:
                final_prediction.loc[df_day.index] = df_day['consommation_mw_moins_7']
                continue

            candidates.sort(key=lambda x: x['distance'])'''

content = re.sub(
    r'\n            candidates\.sort\(key=lambda x: x\[\'distance\'\]\)',
    replacement2,
    content
)

with open('src/models/baselines.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('SUCCESS')
