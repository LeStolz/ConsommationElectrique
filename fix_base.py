
import os
path = 'e:/Projects/Studies/ElectricityConsumption/src/models/baselines.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()
c = c.replace(chr(39) + 'consommation_mw' + chr(39) + '.mean()', chr(39) + 'cible_consommation_mw' + chr(39) + '.mean()')
c = c.replace(chr(39) + 'consommation_mw_moins_7' + chr(39), chr(39) + 'conso_J_moins_7' + chr(39))
c = c.replace(chr(39) + 'consommation_mw_meme_heure_derniere_connue' + chr(39), chr(39) + 'conso_J_moins_1_meme_heure' + chr(39))
c = c.replace(chr(39) + 'consommation_mw' + chr(39) + ')', chr(39) + 'cible_consommation_mw' + chr(39) + ')')
with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

