from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


def get_lagged_local_value(df_target, df_history, lag_days, col):
    """
    Récupère la valeur historique d'une colonne pour un décalage donné en jours,
    en joignant explicitement sur (date_cible - lag, heure locale).
    - Si target et history ont tous les deux 2 heures identiques, on les fait correspondre par leur heure UTC.
    - Sinon, on moyenne les heures doubles de l'historique.
    - Interpole les heures manquantes.
    """
    target_dates = df_target['timestamp_paris'].dt.date - pd.DateOffset(days=lag_days)
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


class Regressor(ABC):
    """
    Interface commune pour tous les modèles de prévision temporelle.
    """

    @abstractmethod
    def fit(self, df_train: pd.DataFrame):
        """
        Entraîne le modèle ou mémorise l'historique nécessaire à la prédiction.
        """
        pass


    @abstractmethod
    def predict(self, df_test: pd.DataFrame) -> pd.Series | np.ndarray:
        """
        Génère les prédictions pour la période de test.
        Doit retourner une Pandas Series ou un array de la même taille que df_test.
        """
        pass

    def get_feature_importances(self) -> pd.DataFrame | None:
        """
        Retourne un DataFrame trié de l'importance des features, ou None si non supporté.
        """
        return None
