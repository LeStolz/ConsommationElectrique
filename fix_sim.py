
import os
path = 'e:/Projects/Studies/ElectricityConsumption/src/models/baselines.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('timestamp_paris', 'timestamp_cible_paris')
c = c.replace(chr(39) + 'temperature_c_pondere_pop' + chr(39), chr(39) + 'meteo_realiste_temperature_c_pondere_pop' + chr(39))

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

