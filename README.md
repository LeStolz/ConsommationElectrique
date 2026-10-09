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

À **14h le jour J**, l'objectif est de prévoir les **24 valeurs horaires de consommation moyenne de la France métropolitaine du jour J+1 (MW)**.

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

L’objectif de la modélisation est d'utiliser les données historiques disponibles afin d’apprendre les comportements passés et de prédire de nouvelles observations, en supposant que les tendances passées restent représentatives du futur, ce qui est vrai dans nos données saisonnière de consommation. Le protocole de modélisation doit ainsi respecter la chronologie des données, en utilisant uniquement les informations disponibles avant les observations à prédire, afin d’éviter toute fuite d’information.

Dans notre cas, la prédiction est réalisée exactement à 14h le jour J. Nous supposons donc qu'à cet instant, les données du jour J ne sont pas encore disponibles, et que seules les données historiques jusqu’au 13h jour J peuvent être utilisées pour construire la prédiction. Cette hypothèse reproduit ainsi les conditions réelles dans lesquelles la prévision serait effectuée.

### 1. Stratégie de prévision

Dans une approche prévision récursive, la prédiction d'une heure est utilisée comme entrée pour prédire l'heure suivante. Dans notre cas, cette approche est problématique car il existe un intervalle entre 14h le jour J et 00h le jour J+1. Il faudrait donc soit traiter sépérament ce cas ou prédire successivement les heures intermédiaires (15h J, 16h J, …, 00h J+1) avant de pouvoir prédire le reste de J+1, ce qui entraînerait une accumulation des erreurs de prédiction.

Nous préfèrons donc de prédire directement les 24 heures de J+1 à partir des informations disponibles jusqu'à 13h le jour J. Pour les modèles capables de gérer plusieurs sorties, un modèle unique est entraîné pour l'ensemble des 24 heures. Cette approche permet de partager une même représentation entre les différentes heures de la journée et peut ainsi exploiter leurs dépendances temporelles.

Les modèles naturellement conçus pour une série temporelle univariée, comme SARIMAX, nécessitent quant à eux une approche récursive : les prédictions intermédiaires sont successivement utilisées pour obtenir les heures suivantes jusqu'à atteindre J+1.

### 2. Feature engineering

Les variables explicatives ont été sélectionnées à partir des résultats de l’analyse exploratoire pour représenter les différentes structures identifiées : dépendances temporelles, saisonnalités, calendrier, météo et évolution du niveau de consommation.

Toutes les variables explicatives sont calculées uniquement à partir des informations disponibles au moment de la prévision, afin d’éviter toute fuite d’information future.

**Calculation des retards temporels en gérant des changements d'heure**. Comme les retards sont calculés à partir de l'heure locale, les changements d'heure doivent être pris en compte :
- **Heure répétée dans les données cibles et historiques :** les occurrences sont appariées à l'aide de leur heure UTC.
- **Heure répétée uniquement dans l'historique :** les valeurs historiques correspondant à la même heure locale sont moyennées.
- **Heures manquantes :** les valeurs sont interpolées linéairement pour compléter les heures manquantes.

#### Variables temporelles

Ces variables permettent de représenter les dépendances temporelles et les variations périodiques de la consommation :
- **Heure, jour de la semaine et jour de l’année :** pour capturer les cycles journaliers, hebdomadaires et annuels mis en évidence par l’ACF.
- **Mois, saisons, saison_météorologue (OHE) :** pour tenir compte des variations saisonnières, notamment liées aux besoins en chauffage et en climatisation.
- **Année :** pour représenter une éventuelle tendance à long terme.
- **Week-end, jours fériés et nombre de zones en vacance :** pour tenir compte des changements de comportement associés aux périodes non travaillées.
- **Numéro de confinement (OHE) :** pour identifier les périodes exceptionnelles susceptibles d’avoir modifié les habitudes de consommation.

#### Encodage cyclique du temps

L’heure, le jour de la semaine et le jour de l’année sont également encodés à l’aide de fonctions sinus et cosinus. Cet encodage préserve la proximité entre les extrémités d’un cycle, comme 23h et 00h, et facilite ainsi l’apprentissage des régularités périodiques par les modèles.

#### Historique de la consommation électrique

Le choix de ces variables repose principalement sur l’analyse ACF/PACF, qui met en évidence les dépendances entre les consommations passées et futures :
- **Consommations décalées d’une semaine (J+1 -7) et d’environ un an (J+1 -365 et J+1 -366) :** pour exploiter les similitudes entre périodes comparables.
- **Dernière consommation connue (13h J) et la consommation à la même heure que l'observation à prédire dernièrement disponible (Xh J si X < 14, sinon Xh J-1) :** pour représenter le niveau récent et les profils horaires habituels.
- **Moyennes sur 24, 48 et 72 heures :** pour caractériser le niveau général de consommation récent.
- **Minimum et maximum sur 24 heures :** pour représenter l’amplitude des variations récentes.
- **Tendances sur 6 et 24 heures :** pour identifier les évolutions récentes.

#### Variables météorologiques

Seules les variables de température point rosée sont retenues, les autres variables météorologiques présentant une faible corrélation observée avec la consommation. La température point rosée peut en effet influencer les besoins en chauffage et en climatisation.

Les variables considérées comprennent :
- **Températures historiques similaire à la consommation (J+1 -7), (J+1 -365), (J+1 -366), (13h J), (Xh J si X < 14, sinon Xh J-1) :** pour exploiter les conditions thermiques passées, notamment à des périodes comparables.
- **Moyennes sur 24, 48 et 72 heures, minimum et maximum sur 24 heures :** pour caractériser le niveau et la variabilité des températures récentes.
- **Tendances sur 6 et 24 heures :** pour représenter leur évolution récente.
- **Indicateurs de froid et de chaleur, cumul des périodes froides sur trois jours et écart à la normale saisonnière :** pour caractériser les conditions thermiques particulières, comme détaillé dans la section d’analyse exploratoire.

### 3. Modèles de référence

Avant de comparer des modèles complexes, plusieurs **baselines** sont utilisées afin de vérifier que les modèles avancés apportent réellement une amélioration.

**Moyenne historique** : Une première référence consiste à prédire chaque heure par la moyenne historique correspondante. Elle fournit un niveau de référence très simple, mais ne tient pas compte des conditions récentes.

**Persistence journalière** : La consommation de J+1 à l'heure $h$ est prédite à partir de la dernière consommation comparable disponible :

$$
\hat{C}_{J+1,h}=C_{J,h} si h < 14, si non \hat{C}_{J+1,h}=C_{J-1,h}
$$

Cette baseline permet de mesurer la difficulté du problème par rapport à une hypothèse de forte persistance à court terme selons ACF, PACF.

**Persistence hebdomadaire** : Utilise la consommation du même jour de la semaine précédente :

$$
\hat{C}_{J+1,h}=C_{J+1 -7,h}
$$

Cette baseline est particulièrement pertinente pour l'électricité, car le comportement de consommation dépend fortement du type de journée selons ACF, PACF.

**Moyenne de jours comparables** : Inspiré par une approache FDA, prédit la consommation du jour J+1 en identifiant les (k) jours historiques les plus similaires au jour J, parmi les même jours de la semaine de quatre semaines précédentes et des périodes comparables de l’année précédente. La similarité repose sur la distance entre les profils de consommation disponibles avant 14h et l’écart entre les températures moyennes correspondantes. La prédiction est ensuite calculée comme une moyenne pondérée des consommations horaires du lendemain de ces jours, en accordant davantage de poids aux plus similaires.

**Régression linéaire simple** : Utilise une régression linéaire avec quelques variables calendaires et retardées. Elle permet d'obtenir un modèle simple et interprétable servant de point de comparaison aux modèles plus complexes.

### 4. Modèles avancés

Les modèles retenus couvrent plusieurs hypothèses complémentaires.

#### 4.1 XGBoost

**Hypothèse :** comme observé lors de l’analyse exploratoire, la consommation présente des effets de seuil et des interactions entre variables (baisse pendant les week-ends et jours fériés, hausse possible lors de températures élevées, etc.). Les modèles à base d’arbres sont adaptés à la capture de ces relations non linéaires, ce qui pourrait améliorer les performances par rapport aux modèles linéaires. De plus, la stabilité relative de la consommation moyenne au cours des dernières années suggère que les prédictions nécessiteront peu d’extrapolation au-delà des valeurs observées à l’entraînement, ce qui constitue un avantage pour ces modèles. Il présente également l'avantage d'être relativement interprétable grâce aux importances de variables et aux méthodes d'explication du modèle.

XGBoost est particulièrement intéressant grâce à son mécanisme de boosting, qui construit successivement des arbres afin de corriger les erreurs des précédents. Cette approche permet de modéliser des interactions complexes entre variables temporelles, historiques et météorologiques.

#### 4.2 Régression linéaire avec termes de Fourier

**Hypothèse :** la consommation électrique peut être prédite en combinant des fonctions Fourier périodiques, qui représentent les cycles récurrents à différentes échelles : journalière, hebdomadaire et annuelle, avec des variables historiques et météorologiques.

Le modèle intègre également les retards de consommation, les variables calendaires et les températures, notamment à travers des indicateurs de froid et de chaleur. Ces derniers permettent de représenter les effets de températures basses ou élevées sur la consommation, dont la relation est linéaire d'après l'analyse exploratoire. Le modèle combine ainsi ces informations dans une relation linéaire pour estimer la consommation future.

Cette approche permet de représenter plusieurs saisonnalités avec un nombre limité de variables. Elle offre ainsi un modèle relativement simple et interprétable.

#### 4.3 Prophet

**Hypothèse :** la consommation électrique peut être représentée comme la combinaison d'une tendance, de plusieurs saisonnalités,  d'effets calendaires et météorologue.

Prophet est un modèle de prévision développé par Meta qui repose sur une décomposition de la série temporelle en plusieurs composantes : une tendance \(g(t)\), des saisonnalités \(s(t)\) et des effets calendaires \(h(t)\), auxquelles s'ajoute une erreur résiduelle.

Prophet est particulièrement pratique, car il intègre nativement la tendance, les saisonnalités et les effets calendaires représentées à l'aide de fonctions de Fourier : il suffit donc d'ajouter les régresseurs météorologiques pertinents.

L'analyse exploratoire met en évidence une saisonnalité journalière, hebdomadaire et annuelle, une évolution du niveau de consommation au cours du temps ainsi que des effets calendaires particuliers. Prophet est donc adapté à ces caractéristiques et fournit une approche complémentaire aux modèles autorégressifs et aux modèles d'apprentissage automatique.

Le **mode additif** est privilégié, car l'amplitude des fluctuations saisonnières apparaît relativement stable.

Sa principale limite réside dans sa capacité à représenter les dépendances complexes entre la consommation future et son historique récent, ainsi que les interactions non linéaires entre la météo et le calendrier. Ces limites justifient sa comparaison avec XGBoost.

#### 4.4 LSTM

**Hypothèse :** la consommation électrique présente des dépendances temporelles et des relations non linéaires avec l'historique, calendrier ou météorologue, que les modèles statistiques ou linéaires ne capturent pas nécessairement. Un réseau LSTM pourrait apprendre ces relations à partir de séquences historiques lui-même (donc pas besoin des variables de retard et les retard comme un ans peut être exprimé par la température).

Le LSTM traite les données sous forme de séquences afin d'apprendre les évolutions de la consommation au cours du temps. L'intérêt du LSTM est sa capacité à apprendre des dépendances temporelles complexes sans imposer explicitement une forme linéaire ou prédéfinie aux relations entre les variables. Toutefois, ses performances dépendent fortement de la fenêtre historique, de l'architecture et des hyperparamètres choisis. Il nécessite également davantage de ressources d'entraînement et est moins interprétable que les modèles précédents.

Le LSTM sera donc évalué afin de déterminer si sa capacité à apprendre des représentations temporelles complexes améliore les performances de prévision par rapport à des approches plus simples, dans les mêmes conditions de validation.

****À AJOUTER DES DÉTAILS****

#### 4.5 SARIMAX

**Hypothèse :** une part importante de la consommation électrique peut être prédite à partir de ses dépendances temporelles passées, en tenant compte la tendance, des saisonnalités, des effets calendaires et des variables météorologiques.

SARIMAX combine des composantes autorégressives, de moyenne mobile, de différenciation et des variables explicatives externes. L'analyse guidera le choix des hyperparamètres (voir `notebooks/03_sarimax`).

Comme il prend en compte nativement des retards sur la consommation, on ajoute que les retards sur un ans.

SARIMAX constitue ainsi un modèle complémentaire à Prophet et XGBoost. Ses principales limites sont la représentation des interactions non linéaires et la prise en compte de plusieurs saisonnalités. Enfin, le modèle devra produire des prévisions multi-pas entre le dernier instant disponible et les heures cibles, ce qui peut entraîner une accumulation d'erreurs.

---

## Protocole expérimental

Afin de garantir une comparaison équitable entre les modèles, un **évaluateur centralisé** assure la préparation des données, la validation temporelle et le calcul des métriques selon un protocole commun.

Les modèles seront évalués selon deux scénarios météorologiques : un scénario **idéal**, dans lequel la météo de l'heure cible \(h\) du jour J+1 est connue au moment de la prévision, servant à établir une borne de performance, et un scénario **réaliste**, dans lequel seules les informations météorologiques disponibles avant 14 h le jour J peuvent être utilisées. L'évaluateur centralisé prendra en charge la préparation des données pour ces deux scénarios.

**Préparation des données et prévention des fuites.** L'évaluateur produit un jeu de données où chaque ligne correspond à une heure cible à prédire et contient les variables explicatives calculées uniquement à partir des informations disponibles avant 14 h le jour J. Chaque modèle utilise exclusivement les variables déjà présentes sur la ligne correspondante pour produire sa prédiction, sans recalculer des informations dans les données historiques, ce qui limite les risques de fuite de données. Lorsque cela est possible, la consommation cible est également masquée (NaN) afin d'éviter toute fuite de données.

**Découpage temporel et réentraînement.** Les données sont réparties chronologiquement en trois périodes afin de préserver l'ordre temporel et de garantir que les ensembles de validation et de test couvrent chacun un cycle annuel complet, permettant ainsi d'évaluer les modèles sur l'ensemble des saisons :

- **Entraînement :** du début de 2016 au début de 2024 (~75 %) ;
- **Validation :** de début 2024 à début 2025 (~5 %) ;
- **Test final :** de début 2025 à septembre 2026 (~10 %).

La validation repose ensuite sur une approche temporelle glissante à pas mensuel, avec les prédiction chaque jour et un réentraînement chaque mois à partir des données historiques disponibles, puis évalué sur le mois suivant. Cette fréquence constitue un compromis entre l'adaptation à l'évolution des comportements de consommation et le coût de calcul. Le test final applique le protocole retenu sans servir à modifier les variables, les hyperparamètres ou le choix du modèle.

Une attention particulière est portée à la frontière entre les ensembles d’entraînement et de validation afin d’éviter toute fuite de données temporelle. Pour une première journée de validation (J+1), l’entraînement ne doit inclure que les données dont la disponibilité est garantie à l’heure de prévision du jour (J), fixée à 14 h. Ainsi, la consommation de la journée (J), dont les valeurs postérieures à 14 h ne sont pas encore connues, ne doit pas être utilisée comme une observation historique complète. La dernière journée entièrement observée intégrée à l’entraînement est donc (J-1).

**Metrique.**
Le choix de la métrique d'erreur dépend de l'application visée. Nous supposons qu'une erreur importante ponctuelle n'est pas nécessairement plus couteux que de petites erreurs récurrentes. En l'absence d'un contexte opérationnel précis, nous privilégions donc la MAE comme métrique principale, et la RMSE comme métrique secondaire. Elles servent à guider la recherche d'hyperparamètres (surapprentissage, régularisation,...) et à évaluer les modèles :
- La capacité des modèles à prévoir les pics de consommation sera évaluée séparément, à travers l'erreur sur la consommation quotidienne totale, erreur sur la valeur de la pointe et erreur sur l’heure de la pointe.
- Enfin, la robustesse des modèles sera étudiée en comparant leurs performances selon différentes conditions : saisons, jours ouvrés et non ouvrés, jours fériés et situations météorologiques. Cette analyse permettra notamment d'identifier les modèles dont les performances restent les plus stables dans des contextes variés.

**Sélection des variables et des hyperparamètres.** La sélection initiale des variables s'appuie sur l'analyse exploratoire et les connaissances du domaine. Leur contribution est ensuite évaluée sur la validation, notamment en comparant les performances avec et sans certains groupes de variables : variables calendaires, variables météorologiques et retards de consommation. Une recherche par grille (*grid search*) permet également d'explorer les hyperparamètres. Les métriques d'entraînement et de validation sont comparées pour détecter un éventuel surapprentissage ou sous-apprentissage.

**Comparaison des modèles.** L'évaluateur calcule les mêmes métriques pour chaque modèle et chaque période, afin d'évaluer l'apport des variables, la stabilité des performances au cours du temps dans différentes saisonnalités et conditions et l'intérêt des modèles plus complexes. Le choix final repose sur les performances de validation, en tenant compte de la robustesse et du coût de calcul. Le jeu de test final est utilisé uniquement pour estimer les performances du modèle retenu sur des données non utilisées pour sa sélection.

## Évaluation











LSTM
plot residues.
ACF residues, Ljung–Box, AIC, maybe use lag features for SARIMAX.
Overfit?

test variable group.
avec critère MAE, RMSE, erreur sur la consommation totale quotidienne, erreur sur la valeur de la pointe et erreur sur l’heure de la pointe ?
Les performances seront également examinées selon les saisons, les jours ouvrés et non ouvrés, les jours fériés ou certaines conditions météorologiques.
Analyser de manière critique les résultats obtenus, les erreurs, prise de recul.

Uncertainty?
On vous donne une série, son ACF/PACF, deux modèles estimés et leurs résidus.
1 La série semble-t-elle stationnaire ? Pourquoi ?
2 Quelle transformation proposeriez-vous si nécessaire ?
3 Quel mécanisme AR/MA/ARMA est plausible ?
4 Quel modèle retenez-vous après estimation ?
5 Les résidus permettent-ils de valider provisoirement ce choix ?

stability in time?
cost?

interface


## 4. Rapport & Restitution
- [ ] Rédiger le rapport synthétique (< 10 pages).
- [ ] Effectuer un **Audit critique de la chaîne de prévision** (1 à 2 pages) :
  - Tableau de disponibilité des informations et risques de fuite.
  - Étude d'ablation / valeur ajoutée de la complexité des modèles.
  - Analyse approfondie d'au moins 3 cas d'échecs majeurs.
- [ ] Documenter l'usage de l'**Agent conversationnel** (3 exemples d'interaction analysés).
- [ ] Préparer la présentation orale (< 10 min) et les diapositives.

- Discussion
- La page de titre, la bibliographie et des annexes techniques raisonnables ne sont pas comptabilisées.

### 8. Audit critique de la chaîne de prévision

2 pages.

- Disponibilité de l’information : Un tableau indiquera, pour chaque variable : sa source, son instant de disponibilité, son caractère observé ou prévu, son utilisation dans le modèle et le risque éventuel de fuite d’information. Il faut répondre à la question : Cette prévision aurait-elle réellement pu être calculée à 14 h le jour J ?
- Valeur ajoutée de la complexité, Cela vaut la peine ? : Une étude d’ablation ou de sensibilité évaluera l’apport de certains groupes de variables, par exemple la météo, le calendrier ou les retards de consommation.
- Analyse des échecs : Au moins trois journées présentant des erreurs importantes seront analysées. Pour chacune, il faut distinguer une limite des données, une limite du modèle, une rupture de régime, un événement difficilement prévisible ou une faiblesse du protocole.
- Robustesse et conditions d’utilisation : Discuter la stabilité du classement des modèles selon les périodes, les métriques et les
heures prévues. Il précisera les situations dans lesquelles il déconseillerait l’utilisation du système proposé.

---

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