import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXRegressor:
    def __init__(self, features_cols, order, seasonal_order, since_year):
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
        self.history = df_train.sort_values(self.date_col).copy()

        if self.since_year is not None:
            self.history = self.history[self.history[self.date_col].dt.year >= self.since_year]

        # On garde une trace de la fin du train pour s'assurer que le predict enchaîne
        self.last_train_date = self.history[self.date_col].max()

        y = self.history[self.target_col].values

        exog = None
        if self.features_cols:
            exog = self.history[self.features_cols].astype(float)

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
            self.model_res = model.fit(disp=False)


    def predict(self, df_test):
        df_pred = df_test.sort_values(self.date_col)

        exog = None
        if self.features_cols:
            exog = df_pred[self.features_cols].astype(float)

        steps = len(df_pred)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model_res.forecast(steps=steps, exog=exog)

        return pd.Series(forecast, index=df_pred.index)
