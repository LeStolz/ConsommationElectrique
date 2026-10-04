from .regressor import Regressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def get_latest_known_value(df, of_col, at_hour):
    df[f'{of_col}_moins_1'] = df[of_col].shift(24)
    df[f'{of_col}_moins_2'] = df[of_col].shift(48)
    before_hour_mask = df['timestamp_paris'].dt.hour <= at_hour
    return np.where(before_hour_mask, df[f'{of_col}_moins_1'], df[f'{of_col}_moins_2'])


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
        combined['consommation_mw_meme_heure_derniere_connue'] = \
            get_latest_known_value(combined, 'consommation_mw', self.hour_of_prediction)

        return combined.loc[df_test.index, 'consommation_mw_meme_heure_derniere_connue']


class LinearRegressor(Regressor):
    """
    Baseline 3 : Régression linéaire simple utilisant le calendrier et la météo,
    et calculant ses propres Lags.
    """
    def __init__(self, hour_of_prediction=14, features_cols=[]):
        self.model = LinearRegression()
        self.hour_of_prediction = hour_of_prediction
        self.features_cols = \
            ['consommation_mw_moins_7', 'consommation_mw_derniere_connue', 'temperature_c_pondere_pop_derniere_connue'] \
            + features_cols


    def _build_features(self, df, history=None):
        if history is not None:
            combined = pd.concat([history, df])
        else:
            combined = df.copy()

        combined['consommation_mw_moins_7'] = combined['consommation_mw'].shift(7 * 24)

        combined['date'] = combined['timestamp_paris'].dt.normalize()
        cutoff = combined.loc[
            combined['timestamp_paris'].dt.hour == self.hour_of_prediction, ['date', 'consommation_mw', 'temperature_c_pondere_pop']
        ].drop_duplicates(subset='date').copy()
        cutoff['date'] += pd.DateOffset(days=1)

        consumption_cutoff = cutoff.set_index('date')['consommation_mw']
        temperature_cutoff = cutoff.set_index('date')['temperature_c_pondere_pop']
        combined['consommation_mw_derniere_connue'] = combined['date'].map(consumption_cutoff)
        combined['temperature_c_pondere_pop_derniere_connue'] = combined['date'].map(temperature_cutoff)

        for col in self.features_cols:
            if col not in combined.columns and '_meme_heure_derniere_connue' in col:
                combined[col] = \
                    get_latest_known_value(combined, col.split('_meme_heure_derniere_connue')[0], self.hour_of_prediction)

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
