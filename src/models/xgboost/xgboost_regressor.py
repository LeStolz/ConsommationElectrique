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


# ============================================================
# PRÉPARATION X / y
# ============================================================

def encoder_categorielles(train, val, test):
    """One-hot encoding sur 'saison' (seule colonne texte), avec un
    encodage appris sur train et appliqué identiquement aux 3 splits
    (mêmes colonnes, même ordre, pour que XGBoost voie une table
    cohérente d'un split à l'autre)."""

    train = pd.get_dummies(train, columns=COLONNES_CATEGORIELLES)
    val = pd.get_dummies(val, columns=COLONNES_CATEGORIELLES)
    test = pd.get_dummies(test, columns=COLONNES_CATEGORIELLES)

    colonnes_saison = [c for c in train.columns if c.startswith("saison_")]
    for df in (val, test):
        for col in colonnes_saison:
            if col not in df.columns:
                df[col] = False

    return train, val, test


def separer_x_y(df, colonnes_features=None):
    """Sépare features (X) et cible (y). Si colonnes_features est None,
    on la déduit automatiquement (toutes les colonnes sauf les
    identifiants/timestamps/cible) -- à appeler d'abord sur train pour
    fixer la liste, puis réutiliser la même liste pour val/test."""

    if colonnes_features is None:
        colonnes_features = [c for c in df.columns if c not in COLONNES_NON_FEATURES]

    # 'weekend' et 'ferie' sont des booléens -> conversion explicite en
    # int pour éviter toute ambiguïté de type avec XGBoost.
    X = df[colonnes_features].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    y = df[COLONNE_CIBLE]

    return X, y, colonnes_features


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def entrainer_xgboost(X_train, y_train, X_val, y_val, **kwargs):
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

    parametres = dict(
        n_estimators=3000,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",  # MAE comme objectif direct
        eval_metric="mae",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    parametres.update(kwargs)

    modele = xgb.XGBRegressor(**parametres)

    modele.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    print(f"\nNombre d'arbres réellement retenus (best_iteration) : "
          f"{modele.best_iteration + 1} / {parametres['n_estimators']}")
    print(f"Meilleure MAE validation : {modele.best_score:.2f}")

    return modele


class XGBoostRegressorCustom(Regressor):
    """
    Modele XGBoost compatible avec l evaluateur universel.
    """
    def __init__(self, scenario="realiste", colonnes_features=None, **kwargs):
        self.scenario = scenario
        self.colonnes_features = colonnes_features
        self.kwargs = kwargs
        self.modele = None

    def fit(self, df_train):
        table_train = df_train.copy()

        # OHE sur saison
        if "saison" in table_train.columns:
            table_train = pd.get_dummies(table_train, columns=COLONNES_CATEGORIELLES)

        if self.colonnes_features is None:
            self.colonnes_features = [c for c in table_train.columns if c not in COLONNES_NON_FEATURES and c != "consommation_mw"]

        # Saisons possibles
        for c in ["saison_Hiver", "saison_Printemps", "saison_Ete", "saison_Automne"]:
            if c in self.colonnes_features and c not in table_train.columns:
                table_train[c] = False

        # Drop cibles manquantes (bord de dataset)
        table_train = table_train.dropna(subset=[COLONNE_CIBLE] + self.colonnes_features)

        X_train = table_train[self.colonnes_features].copy()
        for col in X_train.columns:
            if X_train[col].dtype == bool:
                X_train[col] = X_train[col].astype(int)

        y_train = table_train[COLONNE_CIBLE]

        # Set de validation synthetique (30 derniers jours)
        dates = sorted(table_train["date_prevision"].unique())
        val_dates = set(dates[-30:]) if len(dates) > 30 else set(dates)
        mask_val = table_train["date_prevision"].isin(val_dates)

        X_val = X_train[mask_val].copy()
        y_val = y_train[mask_val].copy()
        X_tr = X_train[~mask_val].copy()
        y_tr = y_train[~mask_val].copy()

        if len(X_tr) == 0:
            X_tr, y_tr = X_val, y_val

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
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=False,
        )

    def predict(self, df_test):
        table_test = df_test.copy()

        if "saison" in table_test.columns:
            table_test = pd.get_dummies(table_test, columns=COLONNES_CATEGORIELLES)

        for c in ["saison_Hiver", "saison_Printemps", "saison_Ete", "saison_Automne"]:
            if c in self.colonnes_features and c not in table_test.columns:
                table_test[c] = False

        X_test = table_test[self.colonnes_features].copy()
        for col in X_test.columns:
            if X_test[col].dtype == bool:
                X_test[col] = X_test[col].astype(int)

        pred = self.modele.predict(X_test)
        return pd.Series(pred, index=df_test.index)

XGBoostRegressor = XGBoostRegressorCustom

# ============================================================
# PRÉDICTION & VÉRIFICATION (Transféré depuis features_xgboost.py)
# ============================================================

def predire_jour_suivant(df, modele, colonnes_features, jour_j, scenario="realiste"):
    """Prédit les 24 (23/25) heures de jour_j+1, sans exiger que ces heures
    existent déjà dans df -- seule l'historique jusqu'à 14h le jour_j est
    nécessaire."""

    df = construire_lags_conso(df)

    df_index = df.set_index(
        [df["timestamp_paris"].dt.normalize(), df["timestamp_paris"].dt.hour]
    )
    df_index.index.names = ["date_paris", "heure_paris"]

    if (jour_j, HEURE_COUPURE) not in df_index.index:
        raise ValueError(f"Pas de ligne (jour_j, 14h) pour {jour_j} dans df.")

    ligne_coupure = df_index.loc[(jour_j, HEURE_COUPURE)]
    if isinstance(ligne_coupure, pd.DataFrame):
        ligne_coupure = ligne_coupure.iloc[0]

    jour_j_plus_1 = jour_j + pd.Timedelta(days=1)

    # On génère nous-mêmes les timestamps cibles -- pas besoin qu'ils
    # existent dans df. tz-aware, gère nativement les jours à 23/25h.
    debut = jour_j_plus_1
    fin = jour_j_plus_1 + pd.Timedelta(days=1)
    timestamps_cibles = pd.date_range(debut, fin, freq="h", tz="Europe/Paris", inclusive="left")

    lignes = []
    for ts in timestamps_cibles:
        horizon_h = ts.hour

        # Lags conso : on les récupère dans l'historique déjà connu, via
        # décalage depuis jour_j_plus_1 (pas besoin que ts existe dans df).
        def lag(colonne, jours):
            cible_lag = (ts - pd.Timedelta(days=jours)).normalize()
            heure_lag = (ts - pd.Timedelta(days=jours)).hour
            if (cible_lag, heure_lag) not in df_index.index:
                return np.nan
            l = df_index.loc[(cible_lag, heure_lag)]
            if isinstance(l, pd.DataFrame):
                l = l.iloc[0]
            return l[colonne]

        ligne = {
            "horizon_h": horizon_h,
            "conso_J_moins_7": lag("consommation_mw", 7),
            "conso_J_moins_365": lag("consommation_mw", 365),
            "conso_J_moins_366": lag("consommation_mw", 366),
            "conso_dernieres_24h_moyenne": ligne_coupure["conso_dernieres_24h_moyenne"],
            "conso_dernieres_24h_min": ligne_coupure["conso_dernieres_24h_min"],
            "conso_dernieres_24h_max": ligne_coupure["conso_dernieres_24h_max"],
            "heure_sin": np.sin(2 * np.pi * horizon_h / 24),
            "heure_cos": np.cos(2 * np.pi * horizon_h / 24),
            "jour_annee_sin": np.sin(2 * np.pi * ts.dayofyear / 365.25),
            "jour_annee_cos": np.cos(2 * np.pi * ts.dayofyear / 365.25),
            "conso_J_moins_1_meme_heure": lag("consommation_mw", 1 if horizon_h <= HEURE_COUPURE else 2),
            "jour_semaine": ts.dayofweek,
            "mois": ts.month,
            "saison": "Hiver" if ts.month in (12, 1, 2) else "Printemps" if ts.month in (3, 4, 5) else "Été" if ts.month in (6, 7, 8) else "Automne",
            "weekend": int(ts.dayofweek >= 5),
            "ferie": 0,       # à compléter via un vrai calendrier de jours fériés si tu en as un
            "vacances": 0,    # idem, calendrier de vacances scolaires
            "confinement_numero": 0,
        }

        for col in COLONNES_METEO:
            ligne[f"meteo_realiste_{col}"] = ligne_coupure[col]

        temp = ligne_coupure["temperature_c_pondere_pop"]
        ligne["degres_sous_15"] = max(0.0, 15 - temp) if pd.notna(temp) else np.nan
        ligne["degres_au_dessus_22"] = max(0.0, temp - 22) if pd.notna(temp) else np.nan

        ligne["timestamp_cible_paris"] = ts
        lignes.append(ligne)

    table = pd.DataFrame(lignes)
    table = pd.get_dummies(table, columns=["saison"])
    for col in colonnes_features:
        if col.startswith("saison_") and col not in table.columns:
            table[col] = False

    X = table[colonnes_features].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    predictions = modele.predict(X)

    return pd.DataFrame({
        "timestamp_cible_paris": table["timestamp_cible_paris"],
        "horizon_h": table["horizon_h"],
        "consommation_predite_mw": predictions,
    }).reset_index(drop=True)

# ============================================================
# VÉRIFICATIONS DE SÉCURITÉ (pas de fuite, pas de ligne incomplète)
# ============================================================

def verifier_table(table):
    """Quelques contrôles de bon sens à lancer après construction."""

    resultats = {}

    resultats["n_lignes"] = len(table)
    resultats["n_jours_prevision"] = table["date_prevision"].nunique()

    # Un jour normal donne 24 lignes, les 2 jours de changement d'heure
    # en donnent 23 ou 25 -- jamais autre chose.
    par_jour = table.groupby("date_prevision").size()
    resultats["repartition_nb_horizons_par_jour"] = par_jour.value_counts().to_dict()
    resultats["jours_hors_23_24_25"] = par_jour[~par_jour.isin([23, 24, 25])].to_dict()

    # La cible ne doit jamais être manquante
    resultats["cible_manquante"] = int(table["cible_consommation_mw"].isna().sum())

    return resultats