from .utils import Regressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class AverageRegressor(Regressor):
    """
    Prédit que la consommation de demain sera exactement identique à la moyenne.
    """
    def fit(self, df_train):
        self.mean = df_train['consommation_mw'].mean()


    def predict(self, df_test):
        return pd.Series(self.mean, index=df_test.index)


class LastWeekPersistenceRegressor(Regressor):
    """
    Prédit que la consommation de demain sera exactement
    identique à celle du même jour de la semaine dernière (J-7).
    """
    def fit(self, df_train):
        pass

    def predict(self, df_test):
        return df_test['conso_J_moins_7']


class YesterdayPersistenceRegressor(Regressor):
    """
    Prédit la consommation de la veille SI elle est connue à 14h,
    sinon se rabat sur l'avant-veille.
    """
    def __init__(self, pred_hour_local=14):
        self.pred_hour_local = pred_hour_local

    def fit(self, df_train):
        pass

    def predict(self, df_test):
        return df_test['conso_J_moins_1_meme_heure']


class LinearRegressor(Regressor):
    """
    Régression linéaire simple utilisant le calendrier et la météo.
    """
    def __init__(self, pred_hour_local=14, model=LinearRegression, features_cols=[]):
        self.model = model()
        self.pred_hour_local = pred_hour_local
        self.features_cols = features_cols

    def fit(self, df_train):
        self.model.fit(df_train[self.features_cols], df_train['consommation_mw'])

    def predict(self, df_test):
        return pd.Series(self.model.predict(df_test[self.features_cols]), index=df_test.index)


class SimilarDayRegressor(Regressor):
    """
    Jours similaires (Même jour de la semaine + Même mois de l'année dernière + Même température).
    """
    def __init__(self, k=4, pred_hour_local=14, temp_weight=1.0):
        self.k = k
        self.pred_hour_local = pred_hour_local
        self.temp_weight = temp_weight


    @staticmethod
    def _normalize(x):
        x = np.asarray(x, dtype=float)
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8: return x - mean
        return (x - mean) / std


    def fit(self, df_train):
        self.history = df_train.tail(366 * 24 + 2).copy()


    def predict(self, df_test):
        assert df_test['timestamp_cible_paris'].dt.date.unique().size <= 1, "Data Leak : Prediction horizon exceeds 1 day."

        date_j_plus_1 = df_test['timestamp_cible_paris'].dt.date.iloc[0]
        date_j = (pd.to_datetime(date_j_plus_1) - pd.Timedelta(days=1)).date()

        hist_dates_date = self.history['timestamp_cible_paris'].dt.date
        day_j = self.history[hist_dates_date == date_j]

        compare_hours = day_j[day_j['timestamp_cible_paris'].dt.hour < self.pred_hour_local]

        # Aligner strictement sur pred_hour_local heures pour gérer les jours de changement d'heure
        s_j = compare_hours.set_index(compare_hours['timestamp_cible_paris'].dt.hour)['consommation_mw']
        s_j = s_j.groupby(s_j.index).mean().reindex(range(self.pred_hour_local)) \
            .interpolate(method='linear', limit_direction='both')

        conso_j = self._normalize(s_j.values)
        mean_temp_j = np.mean(self._normalize(compare_hours['meteo_realiste_temperature_c_pondere_pop'].values))

        date_j_ts = pd.to_datetime(date_j)

        recent_weeks = [(date_j_ts - pd.Timedelta(days=7 * i)).date() for i in range(1, 5)]
        last_year_weeks = [(date_j_ts - pd.Timedelta(days=364 + 7 * i)).date() for i in range(0, 4)]

        hist_unique_dates = set(hist_dates_date)
        candidate_dates = set(recent_weeks + last_year_weeks).intersection(hist_unique_dates)
        candidates = []

        for d in candidate_dates:
            date_d_compare = self.history[
                (hist_dates_date == d) &
                (self.history['timestamp_cible_paris'].dt.hour < self.pred_hour_local)
            ]

            date_d_plus_1_date = (pd.to_datetime(d) + pd.Timedelta(days=1)).date()
            date_d_plus_1 = self.history[hist_dates_date == date_d_plus_1_date]

            s_d = date_d_compare.set_index(date_d_compare['timestamp_cible_paris'].dt.hour)['consommation_mw']
            s_d = s_d.groupby(s_d.index).mean().reindex(range(self.pred_hour_local)) \
                .interpolate(method='linear', limit_direction='both')

            conso_d = self._normalize(s_d.values)
            mean_temp_d = np.mean(self._normalize(date_d_compare['meteo_realiste_temperature_c_pondere_pop'].values))

            dist_shape = np.linalg.norm(conso_j - conso_d)
            dist_temp = abs(mean_temp_j - mean_temp_d)
            total_dist = dist_shape + (self.temp_weight * dist_temp)

            pred_series = date_d_plus_1.groupby(date_d_plus_1['timestamp_cible_paris'].dt.hour)['consommation_mw'].mean()
            candidates.append({
                'distance': total_dist,
                'prediction': pred_series
            })

        candidates.sort(key=lambda x: x['distance'])
        top_k = candidates[:self.k]
        distances = np.array([c['distance'] for c in top_k])

        weights = 1.0 / (distances + 1e-8)
        weights /= weights.sum()

        final_prediction = np.zeros(len(df_test))
        for target_index, hour in enumerate(df_test['timestamp_cible_paris'].dt.hour):
            val = 0
            for c_index, c in enumerate(top_k):
                pred_val = c['prediction'].get(hour, c['prediction'].mean())
                val += pred_val * weights[c_index]
            final_prediction[target_index] = val

        return pd.Series(final_prediction, index=df_test.index)