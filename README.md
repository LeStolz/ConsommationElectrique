# ElectricityConsumption

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