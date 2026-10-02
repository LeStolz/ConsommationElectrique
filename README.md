# ⚡ ElectricityConsumption

Projet de prévision de la consommation électrique horaire en France métropolitaine.

À **14h le jour J**, l'objectif est de prévoir les **24 valeurs horaires de consommation du jour J+1**, en respectant strictly l'information réellement disponible à cet instant (*aucune fuite de données du futur*). 

Le projet couvre la préparation des données, la construction de modèles de référence et avancés, un protocole de validation temporelle rigoureux et un audit critique complet de la chaîne de prévision.

---

## 📊 Données

### État d'avancement

Trois sources de données ont été collectées, nettoyées puis fusionnées en un jeu de données unique au pas horaire, aligné en **UTC**, couvrant la période du **01/01/2016 au 30/09/2026** (**94 224 lignes**).

> 💡 **Principe de reproductibilité** : Toutes les données sont téléchargées manuellement depuis les portails sources et déposées dans `data/raw/`. Elles sont uniquement lues (jamais récupérées dynamiquement via réseau) par les scripts du répertoire `src/pre_processing/`. Aucun appel réseau caché dans le pipeline.

---

### 1. Consommation électrique (RTE / éCO2mix)

Deux fichiers sont nécessaires (téléchargés à la main sur le portail ODRE / OpenDataSoft) car aucun des deux seuls ne couvre l'intégralité de la période **2016 → septembre 2026** :

| Jeu de données | Rôle | Lien | Fichier local |
| :--- | :--- | :--- | :--- |
| **`cons-def`** (consolidées/définitives) | Source principale, la plus fiable, mais s'arrête fin juin 2026 | [ODRE cons-def](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def) | `data/raw/rte_consommation/eco2mix_national_cons_def.csv` |
| **`tr`** (temps réel) | Comble juillet → septembre 2026 ; données encore révisables, donc moins fiables | [ODRE tr](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr) | `data/raw/rte_consommation/eco2mix_national_tr.csv` |

#### Traitement & Nettoyage
* **Pas temporel** : Les deux fichiers sont au pas de 15 minutes (Date et Heure, avec décalage horaire explicite `+01:00`/`+02:00`), avec la colonne `Consommation (MW)` nativement renseignée seulement toutes les 30 minutes (et non 15 — vérifié sur l'ensemble de l'historique 2016-2026) ; agrégation à l'heure par moyenne (`NaN` ignorés automatiquement).
* **Combinaison** : Données définitives prioritaires ; temps réel utilisé seulement pour les heures absentes du fichier définitif (aucun recouvrement écrasé).
* **Traçabilité** : Colonne `source_donnee` conservée (*Données définitives / consolidées / temps réel / interpolé*) pour l'audit de fiabilité — toutes les heures n'ont pas le même niveau de consolidation.
* **Conversion temporelle** : Conversion explicite en UTC (`utc=True`), avec une colonne `timestamp_paris` conservée en plus de `timestamp_utc` pour la traçabilité.
* **Variables calendaires dérivées** : `heure`, `jour_semaine`, `jour_annee` (1 à 365/366 — capture la saisonnalité de façon continue, plus fine que mois ou saison), `mois`, `annee`, `saison`.
* **Imputation** : 10 valeurs manquantes résiduelles interpolées (trous courts) ; **0 valeur manquante restante**.
* **Script** : `src/pre_processing/conso_data_preprocessing.py`

#### Période Covid — décision et vérification

1. **Dates officielles** : Il y a eu 3 confinements nationaux distincts en France (hors couvre-feux et confinements locaux type Dunkerque/Nice) :

| # | Confinement | Début | Fin |
| :-: | :--- | :---: | :---: |
| **1** | Confinement 1 | 17/03/2020 | 11/05/2020 |
| **2** | Confinement 2 | 30/10/2020 | 15/12/2020 |
| **3** | Confinement 3 | 03/04/2021 | 03/05/2021 |

2. **Vérification visuelle sur les données** (`src/pre_processing/verif_periodes_covid.py`, figures dans `reports/figures/covid_zoom_*.png`) : plutôt que de garder les dates officielles sans vérifier, on a tracé la consommation journalière moyenne autour de chaque confinement (±21 jours de référence) pour voir si la rupture dans les données colle bien à ces dates.
   * **Confinement 1** : Rupture nette et bien isolée — la consommation chute et remonte quasiment exactement aux dates officielles (moyenne pendant : **-28,5 %** vs avant). C'est le confinement le plus strict (arrêts d'usines, écoles fermées, télétravail généralisé), donc l'effet est net et peu confondu avec autre chose.
   * **Confinements 2 et 3** : Effet réel mais plus faible et en partie confondu avec la saisonnalité normale (la consommation continue de monter/descendre dans le même sens avant, pendant et après la période rouge, car on entre/sort de l'hiver). Logique : ces confinements étaient moins stricts (écoles ouvertes, plus d'activité économique maintenue).

3. **Décision retenue** : Garder les 3 fenêtres officielles (ce sont les dates des décrets, la référence la plus objective disponible), plutôt qu'une seule grande tranche qui aurait inclus ~13 mois de "normalité" entre les 3 épisodes (été 2020, plusieurs semaines entre chaque confinement) comme période "Covid". Deux variables sont construites :
   * `covid19` (booléen) : `True` sur n'importe laquelle des 3 périodes.
   * `confinement_numero` (`1`, `2`, `3` ou vide) : Permet de distinguer/pondérer différemment les 3 épisodes en modélisation, plutôt que de tout regrouper dans un seul indicateur — utile puisque leur effet sur la consommation n'est pas le même (confinement 1 >> confinements 2 et 3).
   * **Répartition obtenue** : Confinement 1 = 1 343 h, Confinement 2 = 1 128 h, Confinement 3 = 744 h (**3 215 h au total**, sur 94 224 h).

---

### 2. Données météo (SYNOP Météo-France)

#### Source & Téléchargement
Fichiers horaires/3h par station (un fichier par an), téléchargés et placés dans `data/raw/synop_meteo/synop_AAAA.csv` depuis l'URL type :
`https://meteofrance.s3.sbg.io.cloud.ovh.net/data/OBS/SYNOP/synop_AAAA.csv.gz`

#### Stations retenues (1 par région métropolitaine — 13/13 régions)

| Code SYNOP | Station | Région représentée | Population (poids) |
| :---: | :--- | :--- | :---: |
| **7190** | Strasbourg-Entzheim | Grand Est | 5 500 000 |
| **7481** | Lyon-Saint Exupéry | Auvergne-Rhône-Alpes | 8 100 000 |
| **7015** | Lille-Lesquin | Hauts-de-France | 6 000 000 |
| **7149** | Orly | Île-de-France | 12 300 000 |
| **7650** | Marignane | Provence-Alpes-Côte d'Azur | 5 100 000 |
| **7510** | Bordeaux-Mérignac | Nouvelle-Aquitaine | 6 100 000 |
| **7222** | Nantes-Bouguenais | Pays de la Loire | 3 800 000 |
| **7130** | Rennes-Saint Jacques | Bretagne | 3 400 000 |
| **7630** | Toulouse-Blagnac | Occitanie | 6 100 000 |
| **7240** | Tours | Centre-Val de Loire | 2 600 000 |
| **7280** | Dijon-Longvic | Bourgogne-Franche-Comté | 2 800 000 |
| **7027** | Caen-Carpiquet | Normandie | 3 300 000 |
| **7761** | Ajaccio | Corse | 340 000 |

#### Variables extraites & produites (`synop_national_horaire.csv`)

Chaque variable est déclinée en **moyenne simple** et en **moyenne pondérée par la population régionale** (suffixe `_pondere_pop`).

| Colonne (Moyenne simple) | Colonne (Pondérée population) | Description | Unité |
| :--- | :--- | :--- | :---: |
| `temperature_c` | `temperature_c_pondere_pop` | Température | °C |
| `temperature_point_rosee_c` | `temperature_point_rosee_c_pondere_pop` | Point de rosée | °C |
| `humidite_pct` | `humidite_pct_pondere_pop` | Humidité relative | % |
| `vent_direction_deg` | `vent_direction_deg_pondere_pop` | Direction du vent | degrés |
| `vent_vitesse_ms` | `vent_vitesse_ms_pondere_pop` | Vitesse du vent | m/s |
| `nebulosite` | `nebulosite_pondere_pop` | Nébulosité totale | /8 |
| `precip_1h_mm` | `precip_1h_mm_pondere_pop` | Précipitations (dernière heure) | mm/h |

#### Traitement :
* Rééchantillonnage horaire (`resample 1h`), interpolation linéaire (max 3h de trou), puis agrégation spatiale sur les 13 stations.
* **Période** : 01/01/2016 → 01/10/2026 (**94 246 lignes**).

---

### 3. Données calendaires (jours fériés + vacances scolaires)

* **Jours fériés** : Calculés directement avec la librairie Python `holidays` (pas de téléchargement nécessaire).
* **Vacances scolaires** : Téléchargées depuis le [Catalogue Data Éducation](https://data.education.gouv.fr). Fichier déposé dans `data/raw/calendrier/fr-en-calendrier-scolaire.csv`.
  * *Complément 2016-2017* : Ce fichier officiel ne couvrant les zones A/B/C qu'à partir de 2017-2018, la période janvier 2016 → août 2017 a été complétée à la main d'après l'arrêté du 21 janvier 2014 ([Légifrance](https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000028508429)) dans `data/raw/calendrier/vacances_scolaires_2016_2017_complement.csv` (30 lignes, fusionné automatiquement par le script).

#### Variables calendaires générées (`variables_calendaires_horaire.csv`) :

| Colonne | Description |
| :--- | :--- |
| `timestamp_utc` / `timestamp_paris` | Horodatage |
| `heure` | Heure de la journée (0-23, heure de Paris) |
| `jour_semaine` | 0=Lundi … 6=Dimanche |
| `mois` | 1-12 |
| `weekend` | Booléen (Samedi / Dimanche) |
| `nom_ferie` | Nom du jour férié (ou vide) |
| `ferie` | Booléen |
| `vacances` | **Nombre de zones (0 à 3)** en vacances scolaires ce jour-là — pas un simple booléen, car un jour où les 3 zones sont en vacances (ex. Noël) n'a pas le même effet sur la conso nationale qu'un jour où une seule zone l'est (ex. vacances d'hiver décalées par zone) |

* **Période couverte** : 01/01/2016 → 30/09/2026 (**94 224 lignes**).

---

## 🗓️ Roadmap & Prochaines étapes

### 1. Définition & Exploration (EDA)
- [ ] Expliciter la cible, la période d'étude, l'origine de prévision (14h J), l'horizon (24h J+1) et l'information disponible sans fuite de données.
- [ ] Analyser les structures temporelles : cycles horaires, hebdomadaires, saisonnalité annuelle, impact météo, effets calendaires et ruptures (changements d'heure, Covid).

### 2. Modélisation
- [ ] **Modèles de référence (Baselines)** :
  - Persistence naïve : $\hat{C}_{J+1,h} = C_{J,h}$
  - Persistence hebdomadaire : $\hat{C}_{J+1,h} = C_{J-7,h}$
  - Moyenne de jours comparables.
  - Régression linéaire simple (calendrier + retards).
- [ ] **Modèles avancés** :
  - Séries temporelles & régressions sur variables retardées.
  - Modèles avec covariables externes (météo, calendrier).
  - Algorithmes d'apprentissage automatique (XGBoost, LightGBM, Random Forest, etc.).
  - Comparaison entre stratégies de prévision : modèles séparés par heure vs modèle joint.

### 3. Validation & Évaluation
- [ ] Mettre en place un protocole de découpage temporel **Train / Validation / Test** respectant l'ordre chronologique.
- [ ] Simuler strictement les conditions de prévision réelles à 14h.
- [ ] Évaluer la performance globale via **MAE**, **RMSE**, erreur sur la consommation totale quotidienne, erreur sur la pointe (valeur et heure).
- [ ] Segmenter l'évaluation par saison, type de jour (ouvrés/fériés) et conditions météo extrêmes.

### 4. Rapport & Restitution
- [ ] Rédiger le rapport synthétique (< 10 pages).
- [ ] Effectuer un **Audit critique de la chaîne de prévision** (1 à 2 pages) :
  - Tableau de disponibilité des informations et risques de fuite.
  - Étude d'ablation / valeur ajoutée de la complexité des modèles.
  - Analyse approfondie d'au moins 3 cas d'échecs majeurs.
- [ ] Documenter l'usage de l'**Agent conversationnel** (3 exemples d'interaction analysés).
- [ ] Préparer la présentation orale (< 10 min) et les diapositives.

---

## 📂 Structure du projet

```text
├── data/
│   ├── raw/                   # Données brutes déposées manuellement
│   │   ├── rte_consommation/
│   │   ├── synop_meteo/
│   │   └── calendrier/
│   └── processed/             # Données nettoyées, fusionnées et agrégées
├── reports/
│   └── figures/               # Graphiques issus de l'EDA et des évaluations
├── src/
│   ├── pre_processing/        # Scripts de nettoyage et de fusion des données
│   ├── models/                # Entraînement et inférence des modèles
│   └── evaluation/            # Calcul des métriques et analyse d'erreurs
└── README.md