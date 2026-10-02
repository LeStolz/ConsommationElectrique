## Données météo (SYNOP Météo-France)

**Source et téléchargement**

Fichiers horaires/3h par station, un fichier par année, téléchargés à la main (pas d'appel API dans le pipeline) :

```
https://meteofrance.s3.sbg.io.cloud.ovh.net/data/OBS/SYNOP/synop_AAAA.csv.gz
```

À remplacer `AAAA` par chaque année de 2016 à 2026, puis décompresser et placer dans `data/raw/synop_meteo/synop_AAAA.csv`.

Liste officielle des stations (pour vérifier les codes) :
```
https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/postesSynop.csv
```

**Stations retenues (1 par région de métropole, 13/13 régions couvertes)**

| Code SYNOP | Station | Région représentée | Population (poids) |
|---|---|---|---|
| 7190 | Strasbourg-Entzheim | Grand Est | 5 500 000 |
| 7481 | Lyon-Saint Exupéry | Auvergne-Rhône-Alpes | 8 100 000 |
| 7015 | Lille-Lesquin | Hauts-de-France | 6 000 000 |
| 7149 | Orly | Île-de-France | 12 300 000 |
| 7650 | Marignane | Provence-Alpes-Côte d'Azur | 5 100 000 |
| 7510 | Bordeaux-Mérignac | Nouvelle-Aquitaine | 6 100 000 |
| 7222 | Nantes-Bouguenais | Pays de la Loire | 3 800 000 |
| 7130 | Rennes-Saint Jacques | Bretagne | 3 400 000 |
| 7630 | Toulouse-Blagnac | Occitanie | 6 100 000 |
| 7240 | Tours | Centre-Val de Loire | 2 600 000 |
| 7280 | Dijon-Longvic | Bourgogne-Franche-Comté | 2 800 000 |
| 7027 | Caen-Carpiquet | Normandie | 3 300 000 |
| 7761 | Ajaccio | Corse | 340 000 |

Les poids sont des ordres de grandeur (source INSEE), utilisés uniquement pour la moyenne pondérée — pas besoin de plus de précision.

**Variables brutes gardées** (du fichier `synop_AAAA.csv`, séparateur `;`)

| Colonne brute | Signification | Unité brute |
|---|---|---|
| `geo_id_wmo` | identifiant de la station | — |
| `validity_time` | horodatage de l'observation | UTC |
| `t` | température | Kelvin |
| `td` | température du point de rosée | Kelvin |
| `u` | humidité relative | % |
| `dd` | direction du vent | degrés |
| `ff` | vitesse du vent | m/s |
| `n` | nébulosité totale | — |
| `rr1` | précipitations sur la dernière heure | mm |

**Colonnes produites dans `synop_national_horaire.csv`**

Chaque variable existe en deux versions : moyenne simple (chaque station compte pareil) et moyenne pondérée par population régionale (suffixe `_pondere_pop`, plus pertinente pour la conso qui suit surtout les zones peuplées).

| Colonne (moyenne simple) | Colonne (pondérée population) | Description |
|---|---|---|
| `temperature_c` | `temperature_c_pondere_pop` | température (°C) |
| `temperature_point_rosee_c` | `temperature_point_rosee_c_pondere_pop` | point de rosée (°C) |
| `humidite_pct` | `humidite_pct_pondere_pop` | humidité relative (%) |
| `vent_direction_deg` | `vent_direction_deg_pondere_pop` | direction du vent (degrés) |
| `vent_vitesse_ms` | `vent_vitesse_ms_pondere_pop` | vitesse du vent (m/s) |
| `nebulosite` | `nebulosite_pondere_pop` | nébulosité | 
| `precip_1h_mm` | `precip_1h_mm_pondere_pop` | précipitations (mm/h), tronquées à 0 si négatives après interpolation |

Plus `timestamp_utc` et `timestamp_paris`.

**Traitement** : observations ramenées à l'heure (`resample 1h`, interpolation linéaire limitée à 3h de trou), agrégées sur les 13 stations. Période couverte : 2016-01-01 → 2026-10-01 (94 246 lignes).

---

## Données calendaires (jours fériés + vacances scolaires)

**Jours fériés** : calculés directement avec la librairie Python `holidays` (pas de téléchargement nécessaire).

**Vacances scolaires — source et téléchargement**

```
https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/exports/csv?use_labels=true
```

À placer dans `data/raw/calendrier/fr-en-calendrier-scolaire.csv`.

⚠️ Ce fichier officiel ne couvre les zones A/B/C qu'à partir de l'année scolaire 2017-2018. Pour janvier 2016 → août 2017, on a complété à la main à partir de l'arrêté du 21 janvier 2014 (Légifrance, JORFTEXT000028508429) :
```
https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000028508429
```
→ fichier complément `data/raw/calendrier/vacances_scolaires_2016_2017_complement.csv` (30 lignes, fusionné automatiquement par le script s'il est présent).

**Colonnes produites dans `variables_calendaires_horaire.csv`**

| Colonne | Description |
|---|---|
| `timestamp_utc` / `timestamp_paris` | horodatage |
| `heure` | heure de la journée (0-23, heure de Paris) |
| `jour_semaine` | 0=lundi … 6=dimanche |
| `mois` | 1-12 |
| `weekend` | booléen, samedi/dimanche |
| `nom_ferie` | nom du jour férié (ou vide) |
| `ferie` | booléen |
| `vacances` | **nombre de zones (0 à 3)** en vacances scolaires ce jour-là — pas un simple booléen, car un jour où les 3 zones sont en vacances (ex. Noël) n'a pas le même effet sur la conso nationale qu'un jour où une seule zone l'est (ex. vacances d'hiver décalées par zone) |

Période couverte : 2016-01-01 → 2026-09-30 (94 224 lignes).