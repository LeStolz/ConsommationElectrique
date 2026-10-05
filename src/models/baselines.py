from .regressor import Regressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def get_lagged_local_value(df_target, df_history, lag_days, col):
    """
    Récupère la valeur historique d'une colonne pour un décalage donné en jours,
    en joignant explicitement sur (date_cible - lag, heure locale).
    - Moyenne les heures doubles.
    - Interpole les heures manquantes.
    """
    target_dates = df_target['timestamp_paris'].dt.date - pd.Timedelta(days=lag_days)
    target_hours = df_target['timestamp_paris'].dt.hour

    hist_dates = df_history['timestamp_paris'].dt.date
    hist_hours = df_history['timestamp_paris'].dt.hour

    lookup = pd.DataFrame({
        'date': hist_dates,
        'hour': hist_hours,
        'val': df_history[col].values
    })

    # 1. Moyenne des heures doubles (Automne)
    lookup = lookup.groupby(['date', 'hour'], as_index=False)['val'].mean()

    # 2. Interpolation des heures manquantes (Printemps)
    unique_dates = lookup['date'].unique()
    full_index = pd.MultiIndex.from_product(
        [unique_dates, range(24)],
        names=['date', 'hour']
    )

    lookup_series = lookup.set_index(['date', 'hour'])['val']

    # Réindexer sur toutes les heures (0 à 23) pour faire apparaitre les trous (NaN)
    lookup_series = lookup_series.reindex(full_index)

    # Interpoler linéairement (comble le trou de 2h avec la moyenne de 1h et 3h)
    lookup_series = lookup_series.interpolate(method='linear')

    # Mapping final
    keys = pd.MultiIndex.from_arrays([target_dates, target_hours])
    return keys.map(lookup_series).values


def get_latest_local_value(df, col, at_hour_local):
    """
    Récupère la valeur de col à la même heure la veille (J-1),
    ou l'avant-veille (J-2) si on est après 'at_hour'.
    """
    val_j1 = get_lagged_local_value(df, df, lag_days=1, col=col)
    val_j2 = get_lagged_local_value(df, df, lag_days=2, col=col)

    before_hour_mask = df['timestamp_paris'].dt.hour < at_hour_local
    return np.where(before_hour_mask, val_j1, val_j2)


class LastWeekPersistenceRegressor(Regressor):
    """
    Baseline 1 : Prédit que la consommation de demain sera exactement
    identique à celle du même jour de la semaine dernière (J-7).
    """
    def fit(self, df_train):
        self.history = df_train.tail(7 * 24 + 1).copy()


    def predict(self, df_test):
        assert df_test['timestamp_paris'].dt.date.unique().size <= 7, "Data Leak : Prediction horizon exceeds 7 days."
        combined = pd.concat([self.history, df_test])
        pred = get_lagged_local_value(df_test, combined, lag_days=7, col='consommation_mw')
        return pd.Series(pred, index=df_test.index)


class YesterdayPersistenceRegressor(Regressor):
    """
    Baseline 2 : Prédit la consommation de la veille SI elle est connue à 14h,
    sinon se rabat sur l'avant-veille.
    """
    def __init__(self, pred_hour_local=14):
        self.pred_hour_local = pred_hour_local


    def fit(self, df_train):
        self.history = df_train.tail(2 * 24 + 1).copy()


    def predict(self, df_test):
        assert df_test['timestamp_paris'].dt.date.unique().size <= 1, "Data Leak : Prediction horizon exceeds 1 day."
        combined = pd.concat([self.history, df_test])
        pred = get_latest_local_value(combined, 'consommation_mw', self.pred_hour_local)
        return pd.Series(pred, index=df_test.index)


class LinearRegressor(Regressor):
    """
    Baseline 3 : Régression linéaire simple utilisant le calendrier et la météo.
    """
    def __init__(self, pred_hour_local=14, model=LinearRegression, features_cols=[]):
        self.model = model()
        self.pred_hour_local = pred_hour_local
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

        combined['date'] = combined['timestamp_paris'].dt.date
        cutoff = combined.loc[
            combined['timestamp_paris'].dt.hour == self.pred_hour_local - 1, ['date'] + lastest_known_cols
        ].copy()
        cutoff['date'] = (pd.to_datetime(cutoff['date']) + pd.Timedelta(days=1)).dt.date

        for col, lag in last_cols:
            combined[f'{col}_moins_{lag}'] = get_lagged_local_value(combined, combined, lag_days=lag, col=col)

        for col in lastest_known_cols:
            col_cutoff = cutoff.set_index('date')[col]
            combined[f'{col}_derniere_connue'] = combined['date'].map(col_cutoff)

        for col in lastest_known_same_hour_cols:
            combined[f'{col}_meme_heure_derniere_connue'] = get_latest_local_value(combined, col, self.pred_hour_local)

        if history is not None:
            return combined.loc[df.index].copy()
        else:
            return combined.copy()


    def fit(self, df_train):
        self.history = df_train.tail(366 * 24 + 2).copy()
        df_features = self._build_features(df_train)
        self.model.fit(df_features[self.features_cols], df_features['consommation_mw'])


    def predict(self, df_test):
        assert df_test['timestamp_paris'].dt.date.unique().size <= 1, "Data Leak : Prediction horizon exceeds 1 day."
        df_features = self._build_features(df_test, self.history)
        return pd.Series(self.model.predict(df_features), index=df_test.index)


class SimilarDayRegressor(Regressor):
    """
    Baseline 5 : Jours similaires (Même jour de la semaine + Même mois de l'année dernière + Même température).
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
        return (x - mean) / std


    def fit(self, df_train):
        self.history = df_train.tail(366 * 24 + 2).copy()


    def predict(self, df_test):
        assert df_test['timestamp_paris'].dt.date.unique().size <= 1, "Data Leak : Prediction horizon exceeds 1 day."

        date_j_plus_1 = df_test['timestamp_paris'].dt.date.iloc[0]
        date_j = pd.to_datetime(date_j_plus_1 - pd.Timedelta(days=1))

        hist_dates = self.history['timestamp_paris']
        day_j = self.history[hist_dates == date_j]

        compare_hours = day_j[day_j['timestamp_paris'].dt.hour < self.pred_hour_local]

        conso_j = self._normalize(compare_hours['consommation_mw'].values)
        mean_temp_j = np.mean(self._normalize(compare_hours['temperature_c_pondere_pop'].values))

        weekday = date_j.dayofweek
        month = date_j.month
        year = date_j.year

        history_unique_dates = hist_dates.drop_duplicates()
        same_weekdays_dates = history_unique_dates[
            (history_unique_dates.dayofweek == weekday) &
            (history_unique_dates.year == year) &
            (history_unique_dates.month == month) &
            (history_unique_dates.date < date_j.date())
        ]
        same_weekdays_last_year_dates = history_unique_dates[
            (history_unique_dates.dayofweek == weekday) &
            (history_unique_dates.year == year - 1) &
            (history_unique_dates.month == month)
        ]

        candidate_dates = set(same_weekdays_dates.tolist() + same_weekdays_last_year_dates.tolist())
        candidates = []

        for d in candidate_dates:
            date_d = pd.Timestamp(d)
            d_morning = self.history[
                (hist_dates == date_d) &
                (self.history['timestamp_paris'].dt.hour < self.pred_hour_local)
            ]
            d_plus_1 = self.history[hist_dates == date_d + pd.Timedelta(days=1)]

            if len(d_morning) == self.pred_hour_local and len(d_plus_1) > 0:
                if not d_morning['consommation_mw'].isna().any() and not d_plus_1['consommation_mw'].isna().any():
                    conso_c = d_morning['consommation_mw'].values
                    temp_c = d_morning['temperature_c_pondere_pop'].values
                    mean_temp_c = np.mean(temp_c)

                    dist_shape = np.linalg.norm(conso_j - conso_c)
                    dist_temp = abs(mean_temp_j - mean_temp_c)
                    total_dist = dist_shape + (self.temp_weight * dist_temp)

                    pred_series = d_plus_1.groupby(d_plus_1['timestamp_paris'].dt.hour)['consommation_mw'].mean()
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
        for idx, hr in enumerate(df_test['timestamp_paris'].dt.hour):
            val = 0
            for i, c in enumerate(top_k):
                pred_val = c['prediction'].get(hr, c['prediction'].mean())
                val += pred_val * weights[i]
            final_prediction[idx] = val

        return pd.Series(final_prediction, index=df_test.index)
