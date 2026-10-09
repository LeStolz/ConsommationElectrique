import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXRegressor:
    def __init__(self, features_cols=None, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24), since_year=None):
        """
        since_year: SARIMAX est très lent. Il est conseillé de limiter 
        l'historique (ex: since_year=2023) plutôt que d'utiliser des années entières.
        """
        self.features_cols = features_cols if features_cols else []
        self.order = order
        self.seasonal_order = seasonal_order
        self.since_year = since_year
        self.model_res = None
        
        self.date_col = 'cible_timestamp_paris'
        self.target_col = 'cible_consommation_mw'

    def fit(self, df_train):
        # Pour SARIMAX, il faut trier temporellement
        df_fit = df_train.sort_values(self.date_col).copy()
        
        # Tronquer l'historique pour des raisons de performance
        if self.since_year is not None:
            df_fit = df_fit[df_fit[self.date_col].dt.year >= self.since_year]

        # On garde une trace de la fin du train pour s'assurer que le predict enchaîne
        self.last_train_date = df_fit[self.date_col].max()
        
        y = df_fit[self.target_col].values
        
        exog = None
        if self.features_cols:
            exog = df_fit[self.features_cols].astype(float)
            exog = exog.ffill().fillna(0).values

        # On cache les warnings d'optimisation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            # disp=False pour éviter les gros logs
            self.model_res = model.fit(disp=False, maxiter=50)

    def predict(self, df_test):
        df_pred = df_test.sort_values(self.date_col)
        
        exog = None
        if self.features_cols:
            exog = df_pred[self.features_cols].astype(float)
            exog = exog.ffill().fillna(0).values

        steps = len(df_pred)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model_res.forecast(steps=steps, exog=exog)
            
        return pd.Series(forecast, index=df_pred.index)
