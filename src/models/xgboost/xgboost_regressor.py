import pandas as pd
import numpy as np
from xgboost import XGBRegressor

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models.regressor import Regressor
from src.models.xgboost.features_xgboost import construire_table_horizons, predire_jour_suivant
from src.models.xgboost.entrainer_xgboost import entrainer_xgboost, COLONNES_CATEGORIELLES, COLONNES_NON_FEATURES

class XGBoostRegressorCustom(Regressor):
    """
    Modèle XGBoost compatible avec l'évaluateur (TimeSeriesEvaluator).
    """
    def __init__(self, scenario="realiste", colonnes_features=None, **kwargs):
        self.scenario = scenario
        self.colonnes_features = colonnes_features
        self.kwargs = kwargs
        self.modele = None
        self.df_train_raw = None

    def fit(self, df_train):
        self.df_train_raw = df_train.copy()
        
        table_train = construire_table_horizons(df_train, scenario=self.scenario)
        table_train = pd.get_dummies(table_train, columns=COLONNES_CATEGORIELLES)
        
        if self.colonnes_features is None:
            self.colonnes_features = [c for c in table_train.columns if c not in COLONNES_NON_FEATURES]
        
        saisons_possibles = ["saison_Hiver", "saison_Printemps", "saison_Eté", "saison_Automne"]
        for c in saisons_possibles:
            if c in self.colonnes_features and c not in table_train.columns:
                table_train[c] = False

        # --- FIX: Suppression des cibles manquantes ---
        # L'évaluateur censure volontairement l'après-midi du dernier jour de df_train (à 14h)
        # pour éviter les fuites de données. XGBoost ne peut pas s'entraîner sur des cibles NaN.
        mask_notna = table_train["cible_consommation_mw"].notna()
        table_train = table_train[mask_notna].copy()

        X_train = table_train[self.colonnes_features].copy()
        for col in X_train.columns:
            if X_train[col].dtype == bool:
                X_train[col] = X_train[col].astype(int)
                
        y_train = table_train["cible_consommation_mw"]
        
        dates = sorted(table_train["date_prevision"].unique())
        val_dates = set(dates[-30:]) if len(dates) > 30 else set(dates)
        mask_val = table_train["date_prevision"].isin(val_dates)
        
        X_val = X_train[mask_val].copy()
        y_val = y_train[mask_val].copy()
        
        X_tr = X_train[~mask_val].copy()
        y_tr = y_train[~mask_val].copy()
        
        if len(X_tr) == 0:
            X_tr, y_tr = X_val, y_val
            
        self.modele = entrainer_xgboost(X_tr, y_tr, X_val, y_val, **self.kwargs)

    def predict(self, df_test):
        assert df_test['timestamp_paris'].dt.date.unique().size <= 1, "Prediction horizon exceeds 1 day."
        
        jour_j_plus_1 = df_test['timestamp_paris'].dt.normalize().iloc[0]
        jour_j = jour_j_plus_1 - pd.DateOffset(days=1)
        
        df_combined = pd.concat([self.df_train_raw, df_test])
        
        resultat = predire_jour_suivant(
            df=df_combined,
            modele=self.modele,
            colonnes_features=self.colonnes_features,
            jour_j=jour_j,
            scenario=self.scenario
        )
        
        resultat = resultat.set_index("timestamp_cible_paris")
        pred_series = resultat["consommation_predite_mw"].reindex(df_test.index)
        
        return pred_series

XGBoostRegressor = XGBoostRegressorCustom
