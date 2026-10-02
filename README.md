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

> Remplace une première extraction par API (faite par un autre membre du groupe, jugée peu fiable) : l'extraction se fait maintenant exactement comme pour la météo et le calendrier (téléchargement manuel + script de lecture seule).

**2. Météo (SYNOP, Météo-France)**
- Source : exports annuels Météo-France, un fichier par an (`synop_AAAA.csv.gz`), téléchargés à la main sur :
  `https://meteofrance.s3.sbg.io.cloud.ovh.net/data/OBS/SYNOP/synop_AAAA.csv.gz` (ex. `synop_2016.csv.gz` … `synop_2026.csv.gz`)
  → dézippés et déposés dans `data/raw/synop_meteo/`.
- 9 stations retenues pour une couverture représentative du territoire (nord/sud/est/ouest/centre) : Strasbourg, Lyon, Lille, Orly (Paris), Marignane (Marseille), Bordeaux, Nantes, Rennes, Toulouse.
- Variables conservées : température, point de rosée, humidité, vent, nébulosité, précipitations.
- Observations (toutes les 3h) ramenées à l'heure par interpolation linéaire (trous ≤ 3h), puis moyennées entre les 9 stations pour obtenir une série météo nationale.
- Anomalie détectée et corrigée : artefacts d'interpolation produisant de très légères précipitations négatives (14 661 valeurs sur la période étendue) → tronquées à 0.
- Valeurs manquantes résiduelles (trous > 3h, non interpolés) : 107 heures pour température/point de rosée/humidité/vent, 598 pour la nébulosité, 167 pour les précipitations — proportions faibles (< 0,7 %).
- Script : `src/pre_processing/meteo_data_preprocessing.py`

**3. Variables calendaires**
- Heure, jour de la semaine, mois, week-end : dérivés directement du timestamp (aucune source externe nécessaire).
- Jours fériés : calculés avec la bibliothèque `holidays` (pas de téléchargement nécessaire), 121 jours sur 2016-2026.
- Vacances scolaires : fichier téléchargé depuis `data.education.gouv.fr` (dataset `fr-en-calendrier-scolaire`), zones A/B/C combinées (vacances = au moins une zone en congé) ; 1 372 jours en vacances sur 3 927.
- Grille horaire construite en UTC puis convertie en heure de Paris pour les variables calendaires (un bug d'alignement d'1h, dû à une construction initiale en heure de Paris, a été détecté et corrigé avant la fusion).
- Script : `src/pre_processing/calendrier_data_preprocessing.py`

**Fusion**
- Jointure `left` sur `timestamp_utc`, avec la table de consommation comme référence (aucune heure de conso observée n'est perdue).
- 107 heures avec météo manquante après fusion (0,11 %), documentées plutôt que comblées silencieusement (deux trous identifiés, notamment fin août 2017 et mi-avril 2026).
- 0 doublon de `timestamp_utc`.
- Script : `src/pre_processing/fusion_datasets.py`
- Sortie : `data/processed/dataset_final.csv` (94 224 lignes × 21 colonnes)

### Décisions à trancher avant la modélisation

- **Période Covid** : la variable `covid19` est disponible, mais la décision de l'exclure ou non de l'entraînement n'est pas encore prise — à documenter dans le protocole de validation.
- **Fiabilité variable de la consommation récente** : les heures de juillet-septembre 2026 reposent sur des données "temps réel" encore révisables (`source_donnee` = "Données temps réel"), contrairement au reste de la série (consolidé/définitif) — point à traiter explicitement dans l'audit critique (disponibilité de l'information).
- **Valeurs manquantes résiduelles côté météo** (107 à 598 heures selon la variable) : à traiter (interpolation ou exclusion) avant de construire les variables de prévision.
- **Retards de consommation et distinction scénario opérationnel / météo parfaite** : pas encore construits — relèvent de l'étape de modélisation, pas de la préparation des données.

## Contexte

Construire, évaluer et discuter une chaîne complète de prévision de la consommation d’électricité en France métropolitaine.

À 14h le jour J, prévoir la consommation électrique horaire de l’ensemble de la journée J + 1:
$$
\hat{C}_{J+1,h}\ \forall\ h \in \{ 0...23 \}
$$

Le protocole opérationnel doit respecter strictement l’information qui serait réellement disponible à 14 h le jour J.

Un protocole parfait qui utilise a posteriori les infos (météo,...) observées pendant J+1 uniquement comme comparaison ou borne de performance.

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