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
'''
with open('src/models/xgboost/xgboost_regressor.py', 'w', encoding='utf-8') as f:
    f.write(code)
