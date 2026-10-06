import sys
code = '''
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
    Modele XGBoost compatible avec l evaluateur universel.
    """
    def __init__(self, scenario="realiste", colonnes_features=None, **kwargs):
        self.scenario = scenario
        self.colonnes_features = colonnes_features
        self.kwargs = kwargs
        self.modele = None

    def fit(self, df_train):
        # 1. Validation set statique (30 derniers jours)
        dates = sorted(df_train["date_prevision"].unique())
        val_dates = set(dates[-30:]) if len(dates) > 30 else set(dates)
        mask_val = df_train["date_prevision"].isin(val_dates)

        train_split = df_train[~mask_val].copy()
        val_split = df_train[mask_val].copy()

        if len(train_split) == 0:
            train_split = val_split.copy()

        # 2. Encodage categoriel (saison)
        if "saison" in train_split.columns:
            train_split = pd.get_dummies(train_split, columns=COLONNES_CATEGORIELLES)
            val_split = pd.get_dummies(val_split, columns=COLONNES_CATEGORIELLES)

            colonnes_saison = [c for c in train_split.columns if c.startswith("saison_")]
            for df in (val_split,):
                for col in colonnes_saison:
                    if col not in df.columns:
                        df[col] = False

        # Drop NaNs sur la cible
        train_split = train_split.dropna(subset=[COLONNE_CIBLE])
        val_split = val_split.dropna(subset=[COLONNE_CIBLE])

        # 3. Separation X/y
        if self.colonnes_features is None:
            self.colonnes_features = [c for c in train_split.columns if c not in COLONNES_NON_FEATURES]

        def separer(df):
            X = df[self.colonnes_features].copy()
            for col in X.columns:
                if X[col].dtype == bool:
                    X[col] = X[col].astype(int)
            return X, df[COLONNE_CIBLE]

        X_train, y_train = separer(train_split)
        X_val, y_val = separer(val_split)

        # 4. Entrainement
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
            verbose=False,
        )

    def predict(self, df_test):
        test_split = df_test.copy()

        if "saison" in test_split.columns:
            test_split = pd.get_dummies(test_split, columns=COLONNES_CATEGORIELLES)

        for col in self.colonnes_features:
            if col.startswith("saison_") and col not in test_split.columns:
                test_split[col] = False

        X_test = test_split[self.colonnes_features].copy()
        for col in X_test.columns:
            if X_test[col].dtype == bool:
                X_test[col] = X_test[col].astype(int)

        pred = self.modele.predict(X_test)
        return pd.Series(pred, index=df_test.index)

XGBoostRegressor = XGBoostRegressorCustom
'''
with open('src/models/xgboost/xgboost_regressor.py', 'w', encoding='utf-8') as f:
    f.write(code)
