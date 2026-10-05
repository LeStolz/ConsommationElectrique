import pandas as pd
import numpy as np
from prophet import Prophet
from .regressor import Regressor

class ProphetRegressor(Regressor):
    """
    Modèle de prévision utilisant Facebook Prophet.
    """
    def __init__(self, history_days=366, features_cols=[], **prophet_kwargs):
        """
        :param history_days: Nombre de jours d'historique à conserver pour l'entraînement.
        :param features_cols: Liste des colonnes à utiliser comme features supplémentaires (ex: température).
        :param prophet_kwargs: Autres arguments à passer à Prophet (ex: yearly_seasonality=True)
        """
        self.history_days = history_days
        self.features_cols = features_cols
        self.prophet_kwargs = prophet_kwargs
        self.model = None


    def fit(self, df_train):
        if self.history_days is not None:
            cutoff_date = df_train['timestamp_paris'].max() - pd.Timedelta(days=self.history_days)
            df_fit = df_train[df_train['timestamp_paris'] >= cutoff_date].copy()
        else:
            df_fit = df_train.copy()

        df_prophet = pd.DataFrame({
            'ds': df_fit['timestamp_paris'].dt.tz_localize(None),
            'y': df_fit['consommation_mw']
        })

        for feature in self.features_cols:
            df_prophet[feature] = df_fit[feature]

        df_prophet = df_prophet.dropna()

        self.model = Prophet(**self.prophet_kwargs)

        for feature in self.features_cols:
            self.model.add_regressor(feature)

        self.model.fit(df_prophet)


    def predict(self, df_test):
        df_future = pd.DataFrame({
            'ds': df_test['timestamp_paris'].dt.tz_localize(None)
        })

        for feature in self.features_cols:
            df_future[feature] = df_test[feature].values

        forecast = self.model.predict(df_future)

        return pd.Series(forecast['yhat'].values, index=df_test.index)