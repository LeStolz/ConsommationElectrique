from .regressor import Regressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class LastWeekPersistenceRegressor(Regressor):
    """
    Baseline 1 : Prédit que la consommation de demain sera exactement
    identique à celle du même jour de la semaine dernière (J-7).
    """
    def fit(self, df_train):
        self.history = df_train.tail(7 * 24).copy()


    def predict(self, df_test):
        assert len(df_test) <= 7 * 24, "Data Leak : Prediction horizon exceeds 7 days."
        combined = pd.concat([self.history, df_test])
        combined['pred'] = combined['consommation_mw'].shift(7 * 24)
        return combined.loc[df_test.index, 'pred']


class YesterdayPersistenceRegressor(Regressor):
    """
    Baseline 2 : Prédit la consommation de la veille SI elle est connue à 14h,
    sinon se rabat sur l'avant-veille.
    """
    def __init__(self, hour_of_prediction=14):
        self.hour_of_prediction = hour_of_prediction


    def fit(self, df_train):
        self.history = df_train.tail(2 * 24).copy()


    def predict(self, df_test):
        assert len(df_test) <= 24, "Data Leak : Prediction horizon exceeds 24h."
        combined = pd.concat([self.history, df_test])
        combined['conso_J_moins_1'] = combined['consommation_mw'].shift(24)
        combined['conso_J_moins_2'] = combined['consommation_mw'].shift(48)

        before_hour_of_pred_mask = combined['timestamp_paris'].dt.hour < self.hour_of_prediction
        combined['pred'] = np.where(before_hour_of_pred_mask, combined['conso_J_moins_1'], combined['conso_J_moins_2'])

        return combined.loc[df_test.index, 'pred']


class LinearRegressor(Regressor):
    """
    Baseline 3 : Régression linéaire simple utilisant le calendrier et la météo,
    et calculant ses propres Lags en interne.
    """
    def __init__(self, hour_of_prediction=14):
        self.model = LinearRegression()
        self.hour_of_prediction = hour_of_prediction
        self.features_cols = [
            'conso_J_moins_7', 'conso_derniere_connue', 'temperature_c_pondere_pop', 'weekend', 'ferie'
        ]


    def _build_features(self, df, history=None):
        if history is not None:
            combined = pd.concat([history, df])
        else:
            combined = df.copy()

        combined['temperature_c_pondere_pop'] = combined['temperature_c_pondere_pop'].interpolate(method='linear')
        combined['weekend'] = combined['weekend'].ffill()
        combined['ferie'] = combined['ferie'].ffill()

        combined['conso_J_moins_7'] = combined['consommation_mw'].shift(7 * 24)
        combined['conso_J_moins_1'] = combined['consommation_mw'].shift(24)
        combined['conso_J_moins_2'] = combined['consommation_mw'].shift(2 * 24)

        before_hour_of_pred_mask = combined['timestamp_paris'].dt.hour < self.hour_of_prediction
        combined['conso_derniere_connue'] = np.where(
            before_hour_of_pred_mask, combined['conso_J_moins_1'], combined['conso_J_moins_2']
        )

        if history is not None:
            return combined.loc[df.index].copy()
        else:
            return combined.copy()


    def fit(self, df_train):
        # Conserve les 7 derniers jours du train set pour construire les features du test set
        self.history = df_train.tail(7 * 24).copy()

        df_features = self._build_features(df_train)

        # Supprime les NaNs initiaux (les 7 premiers jours du dataset n'ont pas de J-7)
        df_features = df_features.dropna(subset=self.features_cols + ['consommation_mw'])

        self.model.fit(df_features[self.features_cols], df_features['consommation_mw'])


    def predict(self, df_test):
        assert len(df_test) <= 24, "Data Leak : Prediction horizon exceeds 24h."
        # Construit les features du Test en utilisant l'historique pour combler les trous
        df_features = self._build_features(df_test, self.history)
        return self.model.predict(df_features[self.features_cols])
