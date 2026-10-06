
import os
path = 'e:/Projects/Studies/ElectricityConsumption/src/models/prophet.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()
c = c.replace('timestamp_paris', 'timestamp_cible_paris')
c = c.replace(chr(39) + 'consommation_mw' + chr(39), chr(39) + 'cible_consommation_mw' + chr(39))
with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

