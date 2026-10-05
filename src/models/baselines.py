from .regressor import Regressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def get_latest_known_value(df, of_col, at_hour):
    df[f'{of_col}_moins_1'] = df[of_col].shift(24)
    df[f'{of_col}_moins_2'] = df[of_col].shift(48)
    before_hour_mask = df['timestamp_utc'].dt.hour < at_hour
    return np.where(before_hour_mask, df[f'{of_col}_moins_1'], df[f'{of_col}_moins_2'])


class LastWeekPersistenceRegressor(Regressor):
    """
    Baseline 1 : Prédit que la consommation de demain sera exactement
    identique à celle du même jour de la semaine dernière (J-7).
    """
    def fit(self, df_train):
        self.history = df_train.tail(366 * 24).copy()


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
        self.features_cols = features_cols


    def _build_features(self, df, history=None):
        if history is not None:
            combined = pd.concat([history, df])
        else:
            combined = df.copy()

        last_cols = [
            (col.split('_moins_')[0], int(col.split('_moins_')[1]))
            for col in self.features_cols if '_moins_' in col
        ]
        lastest_known_cols = [
            col.split('_derniere_connue')[0]
            for col in self.features_cols if '_derniere_connue' in col and '_meme_heure' not in col
        ]
        lastest_known_same_hour_cols = [
            col.split('_meme_heure_derniere_connue')[0]
            for col in self.features_cols if '_meme_heure_derniere_connue' in col
        ]

        combined['date'] = combined['timestamp_utc'].dt.normalize()
        cutoff = combined.loc[
            combined['timestamp_utc'].dt.hour == self.hour_of_prediction, ['date'] + lastest_known_cols
        ].drop_duplicates(subset='date').copy()
        cutoff['date'] += pd.DateOffset(days=1)

        for col, lag in last_cols:
            combined[f'{col}_moins_{lag}'] = combined[col].shift(lag * 24)

        for col in lastest_known_cols:
            col_cutoff = cutoff.set_index('date')[col]
            combined[f'{col}_derniere_connue'] = combined['date'].map(col_cutoff)

        for col in lastest_known_same_hour_cols:
            combined[f'{col}_meme_heure_derniere_connue'] = get_latest_known_value(combined, col, self.hour_of_prediction)

        if history is not None:
            return combined.loc[df.index].copy()
        else:
            return combined.copy()


    def fit(self, df_train):
        # Conserve les 366 derniers jours du train set pour construire les features du test set
        self.history = df_train.tail(366 * 24).copy()

        df_features = self._build_features(df_train)

        # Supprime les NaNs initiaux (les 7 premiers jours du dataset n'ont pas de J-7)
        df_features = df_features.dropna(subset=self.features_cols + ['consommation_mw'])

        self.model.fit(df_features[self.features_cols], df_features['consommation_mw'])


    def predict(self, df_test):
        assert len(df_test) <= 24, "Data Leak : Prediction horizon exceeds 24h."
        # Construit les features du Test en utilisant l'historique pour combler les trous
        df_features = self._build_features(df_test, self.history)
        return self.model.predict(df_features[self.features_cols])


class SimilarDayRegressor(Regressor):
    """
    Baseline 5 : Jour similaire (`n_candidates` même jour de la semaine + même mois l'année dernière + température).
    """
    def __init__(self, k=4, hour_of_prediction=14, temp_weight=1.0):
        self.k = k
        self.hour_of_prediction = hour_of_prediction
        self.temp_weight = temp_weight


    def _normalize(self, series):
        return (series - series.mean()) / series.std()


    def fit(self, df_train):
        self.history = df_train.tail(2 * 366 * 24).copy()


    def predict(self, df_test):
        assert len(df_test) <= 24, "Data Leak : Prediction horizon exceeds 24h."

        date_j_plus_1 = df_test['timestamp_utc'].dt.normalize().iloc[0]
        date_j = date_j_plus_1 - pd.Timedelta(days=1)
        day_j = self.history[self.history['timestamp_utc'].dt.normalize() == date_j]

        day_j_before_pred = day_j[day_j['timestamp_utc'].dt.hour < self.hour_of_prediction]
        conso_j = self._normalize(day_j_before_pred['consommation_mw'].values)
        mean_temp_j = self._normalize(day_j_before_pred['temperature_c_pondere_pop']).mean()

        weekday = date_j.dayofweek
        month = date_j.month
        year = date_j.year

        history_dates = self.history['timestamp_utc'].dt.normalize().drop_duplicates()
        same_weekday_dates = history_dates[
            (history_dates.dt.dayofweek == weekday) &
            (history_dates.dt.year == year) &
            (history_dates.dt.month == month) &
            (history_dates.dt.date < date_j.date())
        ].tolist()
        same_weekdays_last_year_dates = history_dates[
            (history_dates.dt.dayofweek == weekday) &
            (history_dates.dt.year == year - 1) &
            (history_dates.dt.month == month)
        ].tolist()

        candidate_dates = set(same_weekday_dates + same_weekdays_last_year_dates)
        candidates = []

        for d in candidate_dates:
            d_before_pred = self.history[
                (self.history['timestamp_utc'].dt.normalize() == d) &
                (self.history['timestamp_utc'].dt.hour < self.hour_of_prediction)
            ]
            d_plus_1 = self.history[
                self.history['timestamp_utc'].dt.normalize() == d + pd.Timedelta(days=1)
            ]

            conso_d = self._normalize(d_before_pred['consommation_mw'].values)
            mean_temp_d = self._normalize(d_before_pred['temperature_c_pondere_pop']).mean()

            dist_shape = np.linalg.norm(conso_d - conso_j)
            dist_temp = abs(mean_temp_j - mean_temp_d)
            total_dist = dist_shape + (self.temp_weight * dist_temp)

            candidates.append({
                'distance': total_dist,
                'prediction': d_plus_1['consommation_mw'].values
            })

        candidates.sort(key=lambda x: x['distance'])
        top_k = candidates[:self.k]

        distances = np.array([c['distance'] for c in top_k])
        profiles = np.stack([c['prediction'] for c in top_k])

        weights = 1.0 / (distances + 1e-8)
        weights /= weights.sum()

        final_prediction = np.average(profiles, axis=0, weights=weights)

        return pd.Series(final_prediction, index=df_test.index)