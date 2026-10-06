"""
Construction du tableau de features pour le modèle XGBoost -- scénarios
"météo parfaite" et "météo réaliste".

UN SEUL modèle XGBoost, entraîné sur une table "empilée" où chaque ligne correspond a
 (jour de prévision J, horizon
h de 0 à 23) -> cible = consommation réelle à l'heure h du jour J+1.

Scénario "météo parfaite" (volontairement optimiste, sert de référence
haute) : on utilise la météo RÉELLEMENT OBSERVÉE à l'heure visée (J+1, h)
comme si elle avait été parfaitement prévue à 14h le jour J. C'est une
fuite de données assumée et documentée -- ce scénario ne sera jamais
utilisable en production, seulement comme borne haute de performance pour
juger le scénario "réaliste".

Scénario "météo réaliste" (borne basse honnête) : on n'a pas de vraies
prévisions météo historiques, seulement des observations (synop_national_
horaire.csv). On utilise donc comme proxy de "prévision" la DERNIÈRE VALEUR
MÉTÉO CONNUE À 14H LE JOUR J, persistée telle quelle pour les 24 heures de
J+1 -- même principe que conso_J_moins_1_meme_heure, mais appliqué à la
météo. Aucune fuite de données : à aucun moment on ne regarde une valeur
postérieure à la coupure. C'est une approximation volontairement grossière
(une vraie prévision météo anticiperait un changement de temps, une
persistance non), mais c'est la seule option honnête avec les données
disponibles.
"""




import pandas as pd
import numpy as np
from xgboost import XGBRegressor

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models.utils import Regressor

COLONNE_CIBLE = "cible_consommation_mw"
COLONNES_NON_FEATURES = [
    "date_prevision", "timestamp_cible_paris", "timestamp_cible_utc",
    COLONNE_CIBLE,
]
COLONNES_CATEGORIELLES = ["saison"]


class XGBoostRegressorCustom(Regressor):
    """
    Modèle XGBoost compatible avec l'évaluateur universel.
    """
    def __init__(self, scenario="realiste", colonnes_features=None, **kwargs):
        self.scenario = scenario
        self.colonnes_features = colonnes_features
        self.kwargs = kwargs
        self.modele = None

    def fit(self, df_train):
        """Entraîne un seul modèle XGBoost (option B : horizon_h en feature),
        avec early stopping sur la validation.

        Réglages pensés pour limiter le sur-apprentissage (constaté avec les
        anciens réglages : écart train/val qui s'élargissait au fil des
        arbres -- train MAE 430 vs val MAE 938 à 1500 arbres, contre 672 vs
        999 à 500 arbres -- le train continuait de chuter bien plus vite que
        la validation) :
        - learning_rate abaissé (0.05 -> 0.03) : apprentissage plus lent et
            plus régulier, qui généralise mieux qu'ajouter des arbres au même
            learning_rate.
        - max_depth réduit (6 -> 5) : arbres moins profonds, donc moins
            capables de mémoriser le bruit du train.
        - min_child_weight ajouté (3) : interdit de couper sur des feuilles
            trop petites (= trop spécifiques à quelques lignes du train).
        - reg_lambda (L2) et reg_alpha (L1) ajoutés : pénalisent les poids
            des feuilles, régularisation standard XGBoost.
        - n_estimators remonté à 3000 MAIS avec early_stopping_rounds=50 :
            le plafond est juste une limite de sécurité, c'est l'early
            stopping qui décide du nombre d'arbres réellement utilisé (celui
            qui minimise la MAE validation).
        """

        # 1. Validation set statique (30 derniers jours) pour l'early stopping
        dates = sorted(df_train["date_prevision"].unique())
        val_dates = set(dates[-30:]) if len(dates) > 30 else set(dates)
        mask_val = df_train["date_prevision"].isin(val_dates)

        train_split = df_train[~mask_val].copy()
        val_split = df_train[mask_val].copy()

        if len(train_split) == 0:
            train_split = val_split.copy()

        # 2. Encodage catégoriel (anciennement encoder_categorielles)
        train_split = pd.get_dummies(train_split, columns=COLONNES_CATEGORIELLES)
        val_split = pd.get_dummies(val_split, columns=COLONNES_CATEGORIELLES)

        colonnes_saison = [c for c in train_split.columns if c.startswith("saison_")]
        for df in (train_split, val_split):
            for col in colonnes_saison:
                if col not in df.columns:
                    df[col] = False

        train_split = train_split.dropna(subset=[COLONNE_CIBLE])
        val_split = val_split.dropna(subset=[COLONNE_CIBLE])

        if self.colonnes_features is None:
            self.colonnes_features = [c for c in train_split.columns if c not in COLONNES_NON_FEATURES]

        # 3. Séparation X/y et typage booléen (anciennement separer_x_y)
        def separer(df):
            X = df[self.colonnes_features].copy()
            for col in X.columns:
                if X[col].dtype == bool:
                    X[col] = X[col].astype(int)
            return X, df[COLONNE_CIBLE]

        X_train, y_train = separer(train_split)
        X_val, y_val = separer(val_split)

        # 4. Paramètres et Entraînement (anciennement entrainer_xgboost)
        parametres = dict(
            n_estimators=3000,
            max_depth=5,
            learning_rate=0.03,
            min_child_weight=3,
            reg_lambda=1.0,
            reg_alpha=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:absoluteerror",
            eval_metric="mae",
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1,
        )
        parametres.update(self.kwargs)

        self.modele = XGBRegressor(**parametres)
        self.modele.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )
        print(f"\nNombre d'arbres réellement retenus (best_iteration) : "
            f"{self.modele.best_iteration + 1} / {parametres['n_estimators']}")
        print(f"Meilleure MAE validation : {self.modele.best_score:.2f}")


    def predict(self, df_test):
        test_split = df_test.copy()

        # Encodage pour les données de test
        if "saison" in test_split.columns:
            test_split = pd.get_dummies(test_split, columns=COLONNES_CATEGORIELLES)

        for col in self.colonnes_features:
            if col.startswith("saison_") and col not in test_split.columns:
                test_split[col] = False

        # Extraction des features et typage
        X_test = test_split[self.colonnes_features].copy()
        for col in X_test.columns:
            if X_test[col].dtype == bool:
                X_test[col] = X_test[col].astype(int)

        pred = self.modele.predict(X_test)
        # return pd.DataFrame({
        #     "timestamp_cible_paris": X_test["timestamp_cible_paris"].values,
        #     "horizon_h": X_test["horizon_h"].values,
        #     "consommation_predite_mw": pred,
        # }).sort_values("horizon_h").reset_index(drop=True)
        return pd.Series(pred, index=df_test.index)

XGBoostRegressor = XGBoostRegressorCustom
