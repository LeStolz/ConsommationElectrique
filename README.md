# ElectricityConsumption

Projet de prévision de la consommation électrique horaire en France métropolitaine. À 14h le jour J, l'objectif est de prévoir les 24 valeurs horaires de consommation du jour J+1, en respectant strictement l'information réellement disponible à cet instant (pas de fuite de données du futur). Le projet couvre la préparation des données, la construction de modèles de référence et de modèles plus élaborés, un protocole de validation temporelle rigoureux, et un audit critique de la chaîne de prévision complète.

## Données

### État d'avancement

Trois sources ont été collectées, nettoyées puis fusionnées en un seul jeu de données horaire, aligné en UTC, couvrant **2016-01-01 à 2026-09-30** (94 224 lignes).

> Toutes les données sont téléchargées **à la main** depuis les portails sources et déposées dans `data/raw/`, puis uniquement lues (jamais récupérées) par les scripts de `src/pre_processing/` : aucun appel réseau caché dans le pipeline, tout est reproductible à partir de fichiers versionnés.

**1. Consommation électrique (RTE / éCO2mix)**
- Deux fichiers nécessaires, téléchargés à la main sur le portail ODRE (OpenDataSoft), car aucun des deux seuls ne couvre 2016 → septembre 2026 :
  - **Données consolidées et définitives** (source principale, la plus fiable, mais s'arrête fin juin 2026) :
    https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/export/
    → `data/raw/rte_consommation/eco2mix_national_cons_def.csv`
  - **Données temps réel** (utilisées uniquement pour combler juillet → septembre 2026, absent du fichier définitif ; données encore révisables, donc moins fiables) :
    https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/export/
    → `data/raw/rte_consommation/eco2mix_national_tr.csv`
- Les deux fichiers sont au pas de 15 minutes (`Date et Heure`, avec décalage horaire explicite `+01:00`/`+02:00`) et agrégés à l'heure par moyenne.
- Combinaison : données "définitives" prioritaires ; "temps réel" utilisé seulement pour les heures absentes du fichier définitif (aucun recouvrement écrasé).
- Colonne `source_donnee` conservée (Données définitives / consolidées / temps réel / interpolé) pour l'audit de fiabilité : toutes les heures n'ont pas le même niveau de consolidation.
- Conversion explicite en UTC (`utc=True`), avec une colonne `timestamp_paris` conservée en plus de `timestamp_utc` pour la traçabilité.
- 10 valeurs manquantes résiduelles interpolées (trous courts) ; 0 valeur manquante restante.
- Variable `covid19` (booléen) ajoutée pour isoler la période de forte perturbation (17/03/2020 au 30/06/2021, 11 280 heures concernées).
- Script : `src/pre_processing/conso_data_preprocessing.py`


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

## À faire

### Définition et exploration
- Expliciter le problème (la cible), la période étudiée, l'origine de prévision, l'horizon, l'info dispo, les éventuelles exclusions.

1. 1. Pourquoi ces données

- Consommation électrique issue d'éCO2mix, publiée par RTE.
- Observations météorologiques du réseau SYNOP de Météo-France.
- Variables calendaires construites par le groupe : heure, jour de la semaine, mois, week-end, jours fériés, vacances ou événements justifiés.

Les données seront ramenées à une granularité horaire.

1. 2. L’analyse exploratoire devra faire apparaître les structures pertinentes : cycles horaire, journalier et hebdomadaire, saisonnalité annuelle, relation avec la météo, effets calendaires, ruptures et observations atypiques, changements d'heure.

3. Préparer les données (néttoyage, agrégation, transformation, les stations météorologiques retenues, l’agrégation spatiale, le traitement des valeurs manquantes, des changements d’heure et des observations atypiques) => Les modèles pour ces structures.

### Modèlisation
4. Modèles de référence (justifier) :
	- $\hat{C}_{J+1,h} = C_{J,h}$
	- $\hat{C}_{J+1,h} = C_{J-6/7,h}$
	- Moyenne de plusieurs jours comparables
	- Modèle linéaire simple fondé sur le calendrier et quelques retards.
5. Modèles (chaque méthode doit répondre à une hypothèse ou à une limite
identifiée) :
	- Des modèles de séries temporelles
	- Des régressions sur variables retardées
	- Des méthodes avec covariables externes (météorologiques, calendaires,...)
	- Des méthodes d’apprentissage automatique.
	- Peut-être autres modèles avec meilleur test validation.

	Les 24 heures peuvent être prévues séparément ou conjointement.

### Validation
6. Séparation Train/Validation/Test doit respecter l’ordre chronologique. Le protocole précisera :
	- Les périodes train, validation, test final.
	- La fréquence de réestimation du modèle.
	- La sélection des variables et des hyperparamètres.
	- Les transformations apprises sur les données.
	- La manière de simuler les prévisions produites à 14 h.
	Le jeu de test final ne doit servir ni à choisir les variables, ni à régler les modèles, ni à sélectionner les hyperparamètres.
	Ajuster des modèles.

### Évaluation
Les méthodes seront comparées sur les mêmes dates, la même info dispo,... avec critère MAE, RMSE, erreur sur la consommation totale quotidienne, erreur sur la valeur de la pointe et erreur sur l’heure de la pointe,... autres?

Les performances seront également examinées selon les saisons, les jours ouvrés et non ouvrés, les jours fériés ou certaines conditions météorologiques.

Analyser de manière critique les résultats obtenus, les erreurs, prise de recul.

### Rapport

Il faut expliquer/documenter/interpreter tous dans le rapport (< 10 pages):
- problème,
- choix méthodologiques,
- le protocole d’évaluation,
- les résultats essentiels et l’analyse critique.
- La page de titre, la bibliographie et des annexes techniques raisonnables ne sont pas comptabilisées.
- Les longues portions de code et les sorties non commentées n’ont pas leur place dans le corps du rapport.

#### 8. Audit critique de la chaîne de prévision

1, 2 pages.

- Disponibilité de l’information : Un tableau indiquera, pour chaque variable : sa source, son instant de disponibilité, son caractère observé ou prévu, son utilisation dans le modèle et le risque éventuel de fuite d’information. Il faut répondre à la question : Cette prévision aurait-elle réellement pu être calculée à 14 h le jour J ?
- Valeur ajoutée de la complexité, Cela vaut la peine ? : Une étude d’ablation ou de sensibilité évaluera l’apport de certains groupes de variables, par exemple la météo, le calendrier ou les retards de consommation.
- Analyse des échecs : Au moins trois journées présentant des erreurs importantes seront analysées. Pour chacune, il faut distinguer une limite des données, une limite du modèle, une rupture de régime, un événement difficilement prévisible ou une faiblesse du protocole.
- Robustesse et conditions d’utilisation : Discuter la stabilité du classement des modèles selon les périodes, les métriques et les
heures prévues. Il précisera les situations dans lesquelles il déconseillerait l’utilisation du système proposé.

### Agent conversationnel

Le rapport présentera trois exemples documentés :
1. une proposition d’un agent conservée après vérification ;
2. une proposition modifiée ou rejetée ;
3. une erreur, faiblesse ou réponse trompeuse détectée.
Pour chaque exemple, le groupe indiquera brièvement la tâche demandée, la proposition obtenue, la critique, décision prise et la méthode de vérification. Une annexe peut contenir un journal synthétique des usages transparent.

### À rendre

- Rapport.
- Le code (reproductible) doit couvrir le chargement ou la récupération des données, leur préparation, la construction des variables, l’apprentissage, la prévision, l’évaluation et la production des principaux résultats. Un fichier README précisera les dépendances, l’organisation des fichiers, l’ordre d’exécution et les étapes éventuellement coûteuses.
- Présentation (< 10 mins), accompagnée de diapositives, mettra en avant le problème, le protocole, les principaux choix, les résultats, un cas d’échec significatif et la conclusion critique.