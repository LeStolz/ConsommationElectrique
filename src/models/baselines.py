from .utils import Regressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class AverageRegressor(Regressor):
    """
    Prdit que la consommation de demain sera exactement identique  la moyenne.
    """
    def fit(self, df_train):
        self.mean = df_train['cible_consommation_mw'].mean()


    def predict(self, df_test):
        return pd.Series(self.mean, index=df_test.index)


class LastWeekRegressor(Regressor):
    """
    Prdit que la consommation de demain sera exactement
    identique  celle du mme jour de la semaine dernire (J-7).
    """
    def fit(self, df_train):
        pass

    def predict(self, df_test):
        return df_test['consommation_mw_moins_7']


class YesterdayRegressor(Regressor):
    """
    Prdit la consommation de la veille SI elle est connue  14h,
    sinon se rabat sur l'avant-veille.
    """
    def fit(self, df_train):
        pass

    def predict(self, df_test):
        return df_test['consommation_mw_meme_heure_derniere_connue']


class LinearRegressor(Regressor):
    """
    Rgression linaire simple utilisant le calendrier et la mto.
    """
    def __init__(self, model=LinearRegression, features_cols=[]):
        self.model = model()
        self.features_cols = features_cols
        self.encoded_cols = []

    def _prepare_data(self, df, is_fit=False):
        X = df[self.features_cols].copy()

        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

        if is_fit:
            self.encoded_cols = X.columns.tolist()
        else:
            for col in self.encoded_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.encoded_cols]

        return X

    def fit(self, df_train):
        X_train = self._prepare_data(df_train, is_fit=True)
        self.model.fit(X_train, df_train['cible_consommation_mw'])

    def predict(self, df_test):
        X_test = self._prepare_data(df_test, is_fit=False)
        return pd.Series(self.model.predict(X_test), index=df_test.index)

    def get_feature_importances(self):
        if self.model is None or not hasattr(self.model, 'coef_'):
            return None

        import pandas as pd
        df_imp = pd.DataFrame({
            'Feature': self.encoded_cols,
            'Importance': self.model.coef_
        })
        df_imp['Abs_Importance'] = df_imp['Importance'].abs()
        return df_imp.sort_values(by='Abs_Importance', ascending=False)[['Feature', 'Importance']]


class SimilarDayRegressor(Regressor):
    """
    Jours similaires (Mme jour de la semaine + Mme mois de l'anne dernire + Mme temprature).
    """
    def __init__(self, k=4, pred_hour_local=14, temp_weight=1000.0):
        self.k = k
        self.pred_hour_local = pred_hour_local
        self.temp_weight = temp_weight


    def fit(self, df_train):
        self.history = df_train.tail(2 * 366 * 24).copy()


    def predict(self, df_test):
        final_prediction = pd.Series(index=df_test.index, dtype=float)

        col_temp = "temperature_c_pondere_pop_derniere_connue"

        gap = getattr(self, 'gap_data', pd.DataFrame())
        combined_history = pd.concat([self.history, gap, df_test])

        # Pré-calculer tous les profils journaliers au lieu de faire des masques booléens
        hist_before_14 = combined_history[combined_history['cible_timestamp_paris'].dt.hour < self.pred_hour_local]

        # Profil des heures < 14h
        profiles_14 = hist_before_14.pivot_table(
            index=hist_before_14['cible_timestamp_paris'].dt.date,
            columns=hist_before_14['cible_timestamp_paris'].dt.hour,
            values='cible_consommation_mw',
            aggfunc='mean'
        )

        # Températures moyennes des heures < 14h
        temps_14 = hist_before_14.groupby(hist_before_14['cible_timestamp_paris'].dt.date)[col_temp].mean()

        # Profil complet 24h
        profiles_full = combined_history.pivot_table(
            index=combined_history['cible_timestamp_paris'].dt.date,
            columns=combined_history['cible_timestamp_paris'].dt.hour,
            values='cible_consommation_mw',
            aggfunc='mean'
        )

        hist_unique_dates = set(profiles_14.index)

        for date_j_plus_1 in df_test['cible_timestamp_paris'].dt.date.unique():
            mask_target_day = df_test['cible_timestamp_paris'].dt.date == date_j_plus_1
            df_day = df_test[mask_target_day]

            date_j = (pd.to_datetime(date_j_plus_1) - pd.DateOffset(days=1)).date()

            if date_j not in profiles_14.index:
                final_prediction.loc[df_day.index] = df_day['consommation_mw_meme_heure_derniere_connue']
                continue

            s_j = profiles_14.loc[date_j].reindex(range(self.pred_hour_local)).interpolate(method='linear', limit_direction='both')

            if s_j.isna().all():
                final_prediction.loc[df_day.index] = df_day['consommation_mw_meme_heure_derniere_connue']
                continue

            conso_j = s_j.values
            mean_temp_j = temps_14.get(date_j, np.nan)

            date_j_ts = pd.to_datetime(date_j)
            recent_weeks = [(date_j_ts - pd.DateOffset(days=7 * i)).date() for i in range(1, 5)]
            last_year_weeks = [(date_j_ts - pd.DateOffset(days=364 + 7 * i)).date() for i in range(0, 4)]

            candidate_dates = set(recent_weeks + last_year_weeks).intersection(hist_unique_dates)
            candidates = []

            for d in candidate_dates:
                s_d = profiles_14.loc[d].reindex(range(self.pred_hour_local)).interpolate(method='linear', limit_direction='both')

                if s_d.isna().all():
                    continue

                date_d_plus_1 = (pd.to_datetime(d) + pd.DateOffset(days=1)).date()
                if date_d_plus_1 not in profiles_full.index:
                    continue

                conso_d = s_d.values
                mean_temp_d = temps_14.get(d, np.nan)

                dist_shape = np.linalg.norm(conso_j - conso_d)
                dist_temp = abs(mean_temp_j - mean_temp_d)
                total_dist = dist_shape + (self.temp_weight * dist_temp)

                pred_series = profiles_full.loc[date_d_plus_1]
                candidates.append({
                    'distance': total_dist,
                    'prediction': pred_series
                })

            if len(candidates) == 0:
                final_prediction.loc[df_day.index] = df_day['consommation_mw_meme_heure_derniere_connue']
                continue

            candidates.sort(key=lambda x: x['distance'])
            top_k = candidates[:self.k]
            distances = np.array([c['distance'] for c in top_k])

            weights = 1.0 / (distances + 1e-8)
            weights /= weights.sum()

            for target_index, hour in zip(df_day.index, df_day['cible_timestamp_paris'].dt.hour):
                val = 0
                for c_index, c in enumerate(top_k):
                    pred_val = c['prediction'].get(hour, np.nan)
                    if pd.isna(pred_val): pred_val = c['prediction'].mean()
                    val += pred_val * weights[c_index]
                final_prediction.loc[target_index] = val

        return final_prediction
