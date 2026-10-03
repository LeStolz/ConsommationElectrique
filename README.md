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

À 14h le jour J, prévoir la **average** consommation d’électricité en France métropolitaine horaire de l’ensemble de la journée J + 1:
$$
\hat{C}_{J+1,h}\ \forall\ h \in \{ 0...23 \}
$$
Un protocole parfait qui utilise a posteriori les infos (météo,...) observées pendant J+1 uniquement comme comparaison ou borne de performance.
- Justifier pourquoi ne pas utiliser les autres données dispo de la site :
	- Variables génération are too correlated (e.x if nuclear production drop, some other means might increase) and generation follows demand (e.x. less consommation -> less generation) so using generation is same as consommation.
	- generation depends on weather (solar, wind,...) but we already have weather data which may cover this.
	- Échange is similar to generation.
	- Prix marché is similar in that it is determined by the demand (consommation) AND supply but there is little instant in which supply determines consommation, however, it also introduces noise (e.x if there is an outage, supply spike, but price usually doesn't change immediately). BUT it might reveal info about the covid19.
- Justifier l'utilisation des données consolidées, temps réels, définitifs.
- When covid?
- données ND (non dispo).

- **Période Covid** : la variable `covid19` est disponible, mais la décision de l'exclure ou non de l'entraînement n'est pas encore prise — à documenter dans le protocole de validation on peut utiliser:
	1.  2022-2026
	2.  2019-2026 avec un feature pour covid
	3.  2019-2026 sans covid feature
	4.  2016-2026 avec covid feature
	5.  2016-2026 sans covid feature
	6.  2016-2026 avec covid enlevé


- **Fiabilité variable de la consommation récente** : les heures de juillet-septembre 2026 reposent sur des données "temps réel" encore révisables (`source_donnee` = "Données temps réel"), contrairement au reste de la série (consolidé/définitif) — point à traiter explicitement dans l'audit critique (disponibilité de l'information).
- **Valeurs manquantes résiduelles côté météo** (107 à 598 heures selon la variable) : à traiter (interpolation ou exclusion) avant de construire les variables de prévision.
- **Retards de consommation et distinction scénario opérationnel / météo parfaite** : pas encore construits — relèvent de l'étape de modélisation, pas de la préparation des données.

pourquoi cest donnee
Les données seront ramenées à une granularité horaire.
3. Préparer les données (néttoyage, agrégation, transformation, les stations météorologiques retenues, l’agrégation spatiale, le traitement des valeurs manquantes, des changements d’heure et des observations atypiques) => Les modèles pour ces structures.
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
- Discussion
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
---

- Rapport.
- Le code (reproductible) doit couvrir le chargement ou la récupération des données, leur préparation, la construction des variables, l’apprentissage, la prévision, l’évaluation et la production des principaux résultats. Un fichier README précisera les dépendances, l’organisation des fichiers, l’ordre d’exécution et les étapes éventuellement coûteuses.
- Présentation (< 10 mins), accompagnée de diapositives, mettra en avant le problème, le protocole, les principaux choix, les résultats, un cas d’échec significatif et la conclusion critique.

Météo comment ?
	- Météo prévisé pour J+1
	- Météo J (mais après 14h comment ?)
	- Météo J-1



	- Prophet
	- ATTENTION LE METEO est interpole, alors, attention au fuit de données

Rapport discussion

Le protocole opérationnel doit respecter strictement l’information qui serait réellement disponible à 14 h le jour J.

Un protocole parfait qui utilise a posteriori les infos (météo,...) observées pendant J+1 uniquement comme comparaison ou borne de performance.

Le projet couvre la préparation des données, la construction de modèles de référence et de modèles plus élaborés, un protocole de validation temporelle rigoureux, et un audit critique de la chaîne de prévision complète.


Les 24 heures peuvent être prévues séparément ou conjointement.



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