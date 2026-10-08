# ⚡ ElectricityConsumption

Projet de prévision de la consommation électrique horaire en France métropolitaine.

À **14h le jour J**, l'objectif est de prévoir les **24 valeurs horaires de consommation moyenne du jour J+1**, en respectant strictement l'information réellement disponible à cet instant (*aucune fuite de données du futur*).

Le projet couvre la préparation des données, la construction de modèles de référence et avancés, un protocole de validation temporelle rigoureux et un audit critique complet de la chaîne de prévision.

---

## 📊 Données

Trois sources de données ont été collectées, nettoyées puis fusionnées en un jeu de données unique au pas horaire, aligné en **UTC**, couvrant la période du **01/01/2016 au 30/09/2026** (**94 224 lignes**).

> 💡 **Principe de reproductibilité** : Toutes les données sont téléchargées manuellement depuis les portails sources et déposées dans `data/raw/`. Elles sont uniquement lues (jamais récupérées dynamiquement via réseau) par les scripts du répertoire `src/pre_processing/`. Aucun appel réseau caché dans le pipeline.

---

### 1. Consommation électrique (RTE / éCO2mix)

Deux fichiers sont nécessaires (téléchargés à la main sur le portail ODRE / OpenDataSoft) car aucun des deux seuls ne couvre l'intégralité de la période **2016 → septembre 2026** :

| Jeu de données | Rôle | Lien | Fichier local |
| :--- | :--- | :--- | :--- |
| **`cons-def`** (consolidées/définitives) | Source principale, la plus fiable, mais s'arrête fin juin 2026 | [ODRE cons-def](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def) | `data/raw/rte_consommation/eco2mix_national_cons_def.csv` |
| **`tr`** (temps réel) | Comble juillet → septembre 2026 ; données encore révisables, donc moins fiables | [ODRE tr](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr) | `data/raw/rte_consommation/eco2mix_national_tr.csv` |

Nous avons choisi de ne pas utiliser les autres variables disponibles que les dates et les consommations car elles risquent d'être redondantes avec la consommation :
- Variables production : fortement corrélée entre filières et en partie ajustée à la consommation.
- Variables renouvelables : déjà largement expliquées par les données météorologiques disponibles.
- Variables échanges : similaire aux variables production.
- Variables prix de marché : dépend de l'offre et de la demande et peut donc introduire du bruit comme le prix ne changent pas immédiatement ou du tout même si l'offre ou la demande changent.

#### Traitement & Nettoyage
- **Pas temporel** : Les deux jeux de données sont enregistrés à un pas de 15 minutes (Date et Heure, avec décalage horaire explicite `+01:00`/`+02:00`). Cependant, pour `cons-def`, la variable `Consommation (MW)` n'est renseignée qu'une fois toutes les 30 minutes, les autres observations étant `NaN`. Les données `tr` fournissent quant à elles la consommation toutes les 15 minutes. Comme notre objectif est d'obtenir la **consommation moyenne horaire**, nous agrégeons les observations disponibles par heure en calculant leur moyenne, en ignorant les valeurs `NaN`.
- **Combinaison des sources** : Les données définitives/consolidées sont prioritaires, et les données en temps réel sont utilisées uniquement pour compléter les périodes absentes du fichier, sans écraser les observations disponibles. En effet, les données `tr` ne sont disponibles que sur une période historique limitée et sont progressivement remplacées par les données consolidées puis définitives. Nous ne disposons donc pas d'un historique suffisamment long de données en temps réel (ou consolidées) pour entraîner le modèle à distinguer et apprendre les éventuelles spécificités de ce type de données. Nous acceptons ainsi que notre modèle soit principalement évalué sur des consommations consolidées ou définitives, considérées comme les mesures de référence.
* **Traçabilité** : Colonne `source_donnee` conservée (*Données définitives / consolidées / temps réel / interpolé*) pour l'audit de fiabilité — toutes les heures n'ont pas le même niveau de consolidation.
* **Conversion temporelle** : Conversion explicite en UTC (`utc=True`), avec une colonne `timestamp_paris` conservée en plus de `timestamp_utc` pour la traçabilité.
* **Variables calendaires dérivées** : `heure`, `jour_semaine`, `jour_annee` (1 à 365/366 — capture la saisonnalité de façon continue, plus fine que mois ou saison), `mois`, `annee`, `saison`.
* **Imputation** : 10 valeurs manquantes résiduelles interpolées (trous courts).
* **Script** : `src/pre_processing/conso_data_preprocessing.py`

#### Période Covid — décision et vérification

1. **Dates officielles** : Il y a eu 3 confinements nationaux distincts en France (hors couvre-feux et confinements locaux type Dunkerque/Nice) :

| # | Confinement | Début | Fin |
| :-: | :--- | :---: | :---: |
| **1** | Confinement 1 | 17/03/2020 | 11/05/2020 |
| **2** | Confinement 2 | 30/10/2020 | 15/12/2020 |
| **3** | Confinement 3 | 03/04/2021 | 03/05/2021 |

2. **Vérification visuelle sur les données**

(`src/pre_processing/verif_periodes_covid.py`, figures dans `reports/figures/covid_zoom_*.png`) : plutôt que de garder les dates officielles sans vérifier, on a tracé la consommation journalière moyenne autour de chaque confinement (±21 jours de référence) pour voir si la rupture dans les données colle bien à ces dates.
   * **Confinement 1** : Rupture nette et bien isolée — la consommation chute et remonte quasiment exactement aux dates officielles (moyenne pendant : **-28,5 %** vs avant). C'est le confinement le plus strict (arrêts d'usines, écoles fermées, télétravail généralisé), donc l'effet est net et peu confondu avec autre chose.
   * **Confinements 2 et 3** : Effet réel mais plus faible et en partie confondu avec la saisonnalité normale (la consommation continue de monter/descendre dans le même sens avant, pendant et après la période rouge, car on entre/sort de l'hiver). Logique : ces confinements étaient moins stricts (écoles ouvertes, plus d'activité économique maintenue).

Le comportement de cette période est un peu différent de celui des années normales (figures dans `notebooks/01_exploration_donnees`).

RTE confirme notamment un impact important des confinements sur la consommation électrique. [RTE](https://www.rte-france.com/actualites/mesures-de-deconfinement-la-consommation-en-electricite-reprend-progressivement) indique une baisse pouvant atteindre 20 % au plus fort de la crise.

3. **Décision retenue** :

Nous avons choisi de **conserver la période COVID-19** dans les données, tout en ajoutant un variable afin d'identifier cette période particulière.

Même s'il y a un impact, cette période représente une part relativement limitée de l'ensemble des données. Cependant, la supprimer complètement poserait également un problème pour les **variables retardées**, car elle créerait une rupture dans la série temporelle. De plus, cela réduirait fortement la quantité de données disponibles pour l'entraînement si on ne gardre que les données entre 2022 et 2026. Nous préférons donc conserver cette période et laisser un variable permettre au modèle d'en tenir compte.

On Garde les 3 fenêtres officielles (ce sont les dates des décrets, la référence la plus objective disponible) comme période "Covid". La variable construite est :
   * `confinement_numero` (`1`, `2`, `3` ou vide) : permet de distinguer/pondérer différemment les 3 épisodes en modélisation (et de filtrer "en confinement" avec `confinement_numero.notna()`), plutôt que de tout regrouper dans un seul indicateur booléen — utile puisque leur effet sur la consommation n'est pas le même (confinement 1 >> confinements 2 et 3).
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

**Traitement** : les observations, disponibles toutes les 3 heures, ramenées à l'heure (`resample 1h`), agrégées sur les 13 stations. Entre deux observations, la dernière valeur connue est conservée : par exemple, si une observation est disponible à 15h, sa valeur est utilisée pour 15h, 16h et 17h. Cette méthode permet d'éviter toute fuite d'information, tout en étant adaptée à l'évolution relativement lente des variables météorologiques.

#### Gestion des valeurs manquantes (température et autres variables météo)

Le traitement se fait en 3 situations, dans cet ordre (chaque étape ne traite que ce que la précédente n'a pas réussi à combler) :

1. **Une (ou quelques) station(s) manquante(s), pas toutes** → elle(s) est/sont ignorée(s) dans la moyenne (simple et pondérée par population), qui se recalcule sur les stations restantes. Aucune interpolation n'est nécessaire dans ce cas.
2. **Les 13 stations manquantes en même temps, trou ≤ 3h** → interpolation linéaire (droite entre la valeur juste avant et juste après le trou, limitée à 3h pour rester fiable : sur un trou plus long, une droite ignorerait le cycle jour/nuit).
3. **Les 13 stations manquantes en même temps, trou > 3h** → repli sur la **même heure du jour disponible le plus proche** (la veille si disponible, sinon l'avant-veille, etc), plutôt qu'une interpolation étendue qui inventerait une tendance sur une période trop longue.

Script : `src/pre_processing/meteo_data_preprocessing.py`.

Heures concernées par l'étape 3 (trous > 3h sur les 13 stations simultanément) — deux groupes de trous plus longs identifiés (juillet 2023, avril 2026), le reste sont des trous isolés de quelques heures :

![Heures avec température manquante](../reports/figures/heures_temperature_manquante.png)

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
| `ferie` | Booléen (jour férié) |
| `vacances` | **Nombre de zones (0 à 3)** en vacances scolaires ce jour-là — pas un simple booléen, car un jour où les 3 zones sont en vacances (ex. Noël) n'a pas le même effet sur la conso nationale qu'un jour où une seule zone l'est (ex. vacances d'hiver décalées par zone) |

* **Période couverte** : 01/01/2016 → 30/09/2026 (**94 224 lignes**).

---

## 🎯 Définition

Avec les jeux de données déjà en place, nous redéfinissons notre objectif :

À **14h le jour J**, l'objectif est de prévoir les **24 valeurs horaires de consommation moyenne de la France métropolitaine du jour J+1**.

La période d'étude s'étend du **01/01/2016 au 30/09/2026**. Comme expliqué dans la section consacrée à la période COVID-19, nous avons choisi de **conserver l'ensemble de cette période**, plutôt que de supprimer les années atypiques. Cette durée permet de disposer d'un historique suffisamment long pour exploiter des modèles plus sophistiqués et apprendre les différentes saisonnalités de la consommation, tout en évitant d'intégrer des données trop anciennes qui pourraient être moins représentatives des comportements actuels.

Le modèle doit utiliser uniquement les informations qui auraient réellement été disponibles au moment de la prévision :
- l'historique de consommation connu **jusqu'à 13h le jour J** ;
- les variables calendaires connues à l'avance ;
- les informations météorologiques disponibles **jusqu'à 13h le jour J**, un protocole parfait qui utilise a posteriori ces infos observées pendant J+1 est aussi évalué uniquement comme borne de performance.

### Heure

Toutes les prévisions sont exprimées en **heure locale de Paris**, plutôt qu'en UTC. Ce choix est important car la consommation électrique dépend fortement du rythme de vie de la population, ce qui est lié à des horaires locaux.

La gestion des changements d'heure est également effectuée selon cette logique. Ainsi, lors du passage à l'heure d'hiver, une journée peut comporter deux occurrences de **2h**, tandis que lors du passage à l'heure d'été, l'heure **2h locale peut être absente**. Ces heures correspondent néanmoins au même rythme de vie : elles doivent être interprétées selon leur position dans la journée locale plutôt que simplement selon leur timestamp UTC. Les deux occurrences de 2h lors du passage à l'heure d'hiver doivent donc être prédites comme des heures locales de même nature, tandis que l'heure manquante lors du passage à l'heure d'été ne doit pas être artificiellement créée comme observé lors de l'exploration des données (`notebooks/01_exploration_donnees`).

Les modèles utilisent donc les variables temporelles en **heure locale de Paris** afin d'apprendre les habitudes de consommation selon le rythme journalier, hebdomadaire et saisonnier, tout en conservant l'UTC comme référence technique pour l'alignement des différentes sources de données.

## Exploration

Voir `notebooks/01_exploration_donnees`.

On n'utilise pas FDA parce que :
- À 14h J, on a pas la coubre complète.
- Nos données sont régulières.
- Il y a des trajectoires souvent pas glisses (La consommation connaît de fortes variations entre le matin et le soir)
- Il n'y a pas beaucoup de features, donc le FPCA n'est pas important.

## Modélisation

**Avant de la modéliser**, il faudra donc vérifier la stationnarité de la série et déterminer quelles saisonnalités, variables retardées, informations calendaires et météorologiques sont les plus pertinentes, tout en respectant strictement les informations disponibles à **14h le jour J**.
## Conclusion

Cette analyse exploratoire met en évidence une **forte structure temporelle** de la consommation électrique, avec des variations selon l'heure, le jour de la semaine et la période de l'année. Je soupçonne notamment des **saisonnalités journalière, hebdomadaire et annuelle**, cohérentes avec les profils observés. L'**ACF et la PACF** confirment la présence de dépendances temporelles et permettent d'identifier les retards potentiellement pertinents pour la modélisation.

La série présente également une **tendance à long terme**, confirmée par la décomposition STL et par l'évolution de la moyenne et de l'écart-type. On observe notamment des niveaux et écarts différents **avant, pendant et après la période COVID-19**.

Les variations de température et les variables calendaires apparaissent également comme des facteurs importants. Les **changements d'heure** sont interprétés en heure locale de Paris ; l'effet du changement d'heure lui-même ne semble pas constituer une composante à modéliser séparément.

Ces observations orientent naturellement le choix des modèles : les saisonnalités multiples pourront être exploitées par **XGBoost, Prophet ou une régression avec termes de Fourier**, tandis que les effets spécifiques de la période COVID-19, la calandrier ou la météo pourront être intégrés explicitement, notamment dans les modèles comme **XGBoost ou LSTM**. Mais depuis 2022, le niveau et l'écart est très stable, donc, après l'enlevement de la saisonnalité, la série peut-être stationnaire et **SARIMAX** peut marcher parce qu'il ne demande pas beaucoup de données.

Enfin, les observations atypiques ne seront pas supprimées systématiquement : elles peuvent correspondre à des événements réels tels que des conditions météorologiques extrêmes, des jours fériés ou des périodes exceptionnelles.
STL
yearly trend
transformation, Choisir entre une lecture additive et multiplicative simple.
retard.

Duplicated data at 2h, last sunday of march.
Missing data at 2h, last sunday of oct.
the other one uses meteo J-1 meme h if possible, if not J-2 meme h

Our models will use direct inference because recursive inference is a bit hard with the first prediction (hour 0) having no previous hour to use for it (it only has J-1 14h and 0h) unlike others (which has J-1 14h, hh and h-1h) which complicate things.

However, for direct inference, we will be using only 1 model to learn the entire 24 hours as the time of training is very long and also because there are correlations between hours, 1 model might be able to learn that correlation.
many params.

Models will be retrained every month because consumption pattern do not change that quickly and some models take very long to train. Within that month, the models don't need to use recursive because we get the real values right away.

SARIMAX => modifié/limites, stationarity, diff + diff saison, correction saison ? transformation ? justifier avec ACF, PACF <-> candidats, plot residues, ACF residues, Ljung–Box.

XGBoost => Bon
- Feature engineering C[J], C[J-7], C[J-365], C[J-366], T[J], T[J-7],...
- Unstable, overfit?
- Intepretable
LSTM => Bon, T[J, <=14h], T[J-1], T[J-7], T[J-365], T[J-366]
résidu.
Prophet par Meta
Regression => Linear / Fourier.
4. Modèles de référence (justifier) :
	- moyenne historique
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
```

---

Uncertainty?
On vous donne une série, son ACF/PACF, deux modèles estimés et leurs résidus.
1 La série semble-t-elle stationnaire ? Pourquoi ?
2 Quelle transformation proposeriez-vous si nécessaire ?
3 Quel mécanisme AR/MA/ARMA est plausible ?
4 Quel modèle retenez-vous après estimation ?
5 Les résidus permettent-ils de valider provisoirement ce choix ?

AIC/BIC
stability in time?
cost?

interface

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
Les méthodes seront comparées sur les mêmes dates, la même info dispo,... avec critère MAE, RMSE, erreur sur la consommation totale quotidienne, erreur sur la valeur de la pointe et erreur sur l’heure de la pointe ?
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