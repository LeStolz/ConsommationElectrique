from .regressor import Regressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def get_lagged_local_value(df_target, df_history, lag_days, col):
    """
    Récupère la valeur historique d'une colonne pour un décalage donné en jours,
    en joignant explicitement sur (date_cible - lag, heure locale).
    - Si target et history ont tous les deux 2 heures identiques, on les fait correspondre par leur heure UTC.
    - Sinon, on moyenne les heures doubles de l'historique.
    - Interpole les heures manquantes.
    """
    target_dates = df_target['timestamp_paris'].dt.date - pd.Timedelta(days=lag_days)
    target_hours = df_target['timestamp_paris'].dt.hour
    target_utc_hours = df_target['timestamp_utc'].dt.hour

    hist_dates = df_history['timestamp_paris'].dt.date
    hist_hours = df_history['timestamp_paris'].dt.hour
    hist_utc_hours = df_history['timestamp_utc'].dt.hour

    hist_df = pd.DataFrame({
        'date': hist_dates,
        'hour': hist_hours,
        'utc_hour': hist_utc_hours,
        'val': df_history[col].values
    })

    # Identifier les doublons dans l'historique
    hist_df['hist_count'] = hist_df.groupby(['date', 'hour'])['date'].transform('count')

    # 1. Lookup Exact en utilisant UTC
    lookup_exact = hist_df.set_index(['date', 'hour', 'utc_hour'])['val']

    # 2. Lookup Moyen
    hist_mean = hist_df.groupby(['date', 'hour'], as_index=False)['val'].mean()
    unique_dates = hist_mean['date'].unique()
    full_index = pd.MultiIndex.from_product([unique_dates, range(24)], names=['date', 'hour'])
    lookup_mean = hist_mean.set_index(['date', 'hour'])['val'].reindex(full_index)
    lookup_mean = lookup_mean \
        .groupby(level='date') \
        .transform(lambda s: s.interpolate(method='linear', limit_direction='both'))

    target_df = pd.DataFrame({
        'date': target_dates,
        'hour': target_hours,
        'utc_hour': target_utc_hours
    })
    target_df['target_count'] = target_df.groupby(['date', 'hour'])['date'].transform('count')

    hist_counts_series = hist_df.groupby(['date', 'hour'])['hist_count'].first()
    target_df['hist_count'] = pd.MultiIndex.from_arrays([target_df['date'], target_df['hour']]).map(hist_counts_series)

    use_exact_mask = (target_df['target_count'] == 2) & (target_df['hist_count'] == 2)

    keys_mean = pd.MultiIndex.from_arrays([target_df['date'], target_df['hour']])
    keys_exact = pd.MultiIndex.from_arrays([target_df['date'], target_df['hour'], target_df['utc_hour']])

    vals_mean = keys_mean.map(lookup_mean).values
    vals_exact = keys_exact.map(lookup_exact).values

    return np.where(use_exact_mask, vals_exact, vals_mean)


def get_latest_local_value(df_target, df_history, col, at_hour_local):
    """
    Récupère la valeur de col à la même heure la veille (J-1),
    ou l'avant-veille (J-2) si on est après 'at_hour'.
    """
    val_j1 = get_lagged_local_value(df_target, df_history, lag_days=1, col=col)
    val_j2 = get_lagged_local_value(df_target, df_history, lag_days=2, col=col)

    before_hour_mask = df_target['timestamp_paris'].dt.hour < at_hour_local
    return np.where(before_hour_mask, val_j1, val_j2)


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
        self.history = df_train.tail(7 * 24 + 1).copy()


    def predict(self, df_test):
        assert df_test['timestamp_paris'].dt.date.unique().size <= 7, "Data Leak : Prediction horizon exceeds 7 days."
        combined = pd.concat([self.history, df_test])
        pred = get_lagged_local_value(df_test, combined, lag_days=7, col='consommation_mw')
        return pd.Series(pred, index=df_test.index)


class YesterdayPersistenceRegressor(Regressor):
    """
    Prédit la consommation de la veille SI elle est connue à 14h,
    sinon se rabat sur l'avant-veille.
    """
    def __init__(self, pred_hour_local=14):
        self.pred_hour_local = pred_hour_local


    def fit(self, df_train):
        self.history = df_train.tail(2 * 24 + 1).copy()


    def predict(self, df_test):
        assert df_test['timestamp_paris'].dt.date.unique().size <= 1, "Data Leak : Prediction horizon exceeds 1 day."
        combined = pd.concat([self.history, df_test])
        pred = get_latest_local_value(df_test, combined, 'consommation_mw', self.pred_hour_local)
        return pd.Series(pred, index=df_test.index)


class LinearRegressor(Regressor):
    """
    Régression linéaire simple utilisant le calendrier et la météo.
    """
    def __init__(self, pred_hour_local=14, model=LinearRegression, features_cols=[]):
        self.model = model()
        self.pred_hour_local = pred_hour_local
        self.features_cols = features_cols
        self._cached_features = None


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
            combined[f'{col}_meme_heure_derniere_connue'] = get_latest_local_value(combined, combined, col, self.pred_hour_local)

        if history is not None:
            return combined.loc[df.index].copy()
        else:
            return combined.copy()


    def fit(self, df_train):
        self.history = df_train.tail(366 * 24 + 2).copy()

        if self._cached_features is None:
            self._cached_features = self._build_features(df_train)
        else:
            new_idx = df_train.index.difference(self._cached_features.index)
            if len(new_idx) > 0:
                new_rows = df_train.loc[new_idx]
                needed_history = df_train.loc[~df_train.index.isin(new_idx)].tail(366 * 24 + 2)
                new_features = self._build_features(new_rows, history=needed_history)
                self._cached_features = pd.concat([self._cached_features, new_features])

        df_features = self._cached_features.loc[df_train.index]
        df_features = df_features.dropna(subset=self.features_cols + ['consommation_mw'])
        self.model.fit(df_features[self.features_cols], df_features['consommation_mw'])


    def predict(self, df_test):
        assert df_test['timestamp_paris'].dt.date.unique().size <= 1, "Data Leak : Prediction horizon exceeds 1 day."
        df_features = self._build_features(df_test, self.history)
        return pd.Series(self.model.predict(df_features[self.features_cols]), index=df_test.index)


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
        assert df_test['timestamp_paris'].dt.date.unique().size <= 1, "Data Leak : Prediction horizon exceeds 1 day."

        date_j_plus_1 = df_test['timestamp_paris'].dt.date.iloc[0]
        date_j = (pd.to_datetime(date_j_plus_1) - pd.Timedelta(days=1)).date()

        hist_dates_date = self.history['timestamp_paris'].dt.date
        day_j = self.history[hist_dates_date == date_j]

        compare_hours = day_j[day_j['timestamp_paris'].dt.hour < self.pred_hour_local]

        # Aligner strictement sur pred_hour_local heures pour gérer les jours de changement d'heure
        s_j = compare_hours.set_index(compare_hours['timestamp_paris'].dt.hour)['consommation_mw']
        s_j = s_j.groupby(s_j.index).mean().reindex(range(self.pred_hour_local)) \
            .interpolate(method='linear', limit_direction='both')

        conso_j = self._normalize(s_j.values)
        mean_temp_j = np.mean(self._normalize(compare_hours['temperature_c_pondere_pop'].values))

        date_j_ts = pd.to_datetime(date_j)

        recent_weeks = [(date_j_ts - pd.Timedelta(days=7 * i)).date() for i in range(1, 5)]
        last_year_weeks = [(date_j_ts - pd.Timedelta(days=364 + 7 * i)).date() for i in range(0, 4)]

        hist_unique_dates = set(hist_dates_date)
        candidate_dates = set(recent_weeks + last_year_weeks).intersection(hist_unique_dates)
        candidates = []

        for d in candidate_dates:
            date_d_compare = self.history[
                (hist_dates_date == d) &
                (self.history['timestamp_paris'].dt.hour < self.pred_hour_local)
            ]

            date_d_plus_1_date = (pd.to_datetime(d) + pd.Timedelta(days=1)).date()
            date_d_plus_1 = self.history[hist_dates_date == date_d_plus_1_date]

            s_d = date_d_compare.set_index(date_d_compare['timestamp_paris'].dt.hour)['consommation_mw']
            s_d = s_d.groupby(s_d.index).mean().reindex(range(self.pred_hour_local)) \
                .interpolate(method='linear', limit_direction='both')

            conso_d = self._normalize(s_d.values)
            mean_temp_d = np.mean(self._normalize(date_d_compare['temperature_c_pondere_pop'].values))

            dist_shape = np.linalg.norm(conso_j - conso_d)
            dist_temp = abs(mean_temp_j - mean_temp_d)
            total_dist = dist_shape + (self.temp_weight * dist_temp)

            pred_series = date_d_plus_1.groupby(date_d_plus_1['timestamp_paris'].dt.hour)['consommation_mw'].mean()
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
        for target_index, hour in enumerate(df_test['timestamp_paris'].dt.hour):
            val = 0
            for c_index, c in enumerate(top_k):
                pred_val = c['prediction'].get(hour, c['prediction'].mean())
                val += pred_val * weights[c_index]
            final_prediction[target_index] = val

        return pd.Series(final_prediction, index=df_test.index)