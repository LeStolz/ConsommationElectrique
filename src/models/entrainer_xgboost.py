"""
Entraînement et évaluation du modèle XGBoost -- scénario météo parfaite.

Stratégie (option B, décidée avec l'équipe) : un seul modèle XGBoost,
entraîné sur la table empilée (jour de prévision x horizon_h), avec
`horizon_h` comme feature parmi les autres.

Ce script ne fait QUE la partie modélisation -- la construction de la
table de features est dans features_xgboost.py, déjà exécutée et
sauvegardée en CSV par le notebook.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

COLONNE_CIBLE = "cible_consommation_mw"

# Colonnes qui ne doivent JAMAIS servir de features (identifiants,
# timestamps, cible elle-même).
COLONNES_NON_FEATURES = [
    "date_prevision", "timestamp_cible_paris", "timestamp_cible_utc",
    COLONNE_CIBLE,
]

# Colonnes catégorielles à encoder avant de donner la table à XGBoost.
COLONNES_CATEGORIELLES = ["saison"]


# ============================================================
# CHARGEMENT DES SPLITS DÉJÀ CONSTRUITS
# ============================================================

def charger_splits(dossier=None):
    dossier = dossier or PROCESSED_DIR
    train = pd.read_csv(dossier / "features_xgboost_parfait_train.csv")
    val = pd.read_csv(dossier / "features_xgboost_parfait_val.csv")
    test = pd.read_csv(dossier / "features_xgboost_parfait_test.csv")
    return train, val, test


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


# ============================================================
# ÉVALUATION
# ============================================================

def evaluer(modele, X, y, horizon_h=None):
    """MAE/RMSE/MAPE globaux. Si horizon_h (la colonne du même nom dans X
    ou une Series externe) est fourni, renvoie aussi le détail par
    horizon."""

    pred = modele.predict(X)

    resultats = {
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(mean_squared_error(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / y)) * 100),
    }

    if horizon_h is not None:
        df_eval = pd.DataFrame({"horizon_h": horizon_h.values, "y": y.values, "pred": pred})
        par_horizon = df_eval.groupby("horizon_h").apply(
            lambda g: pd.Series({
                "mae": mean_absolute_error(g["y"], g["pred"]),
                "rmse": np.sqrt(mean_squared_error(g["y"], g["pred"])),
            }),
            include_groups=False,
        )
        resultats["par_horizon"] = par_horizon

    return resultats, pred


def importance_features(modele, colonnes_features, top_n=20):
    importances = pd.Series(modele.feature_importances_, index=colonnes_features)
    return importances.sort_values(ascending=False).head(top_n)