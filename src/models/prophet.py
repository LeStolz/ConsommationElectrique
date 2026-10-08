import pandas as pd
import numpy as np
import logging
import contextlib
import os
from prophet import Prophet
from .utils import Regressor

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


class ProphetRegressor(Regressor):
    """
    Modle de prvision utilisant Facebook Prophet, optimis pour l'lectricit.
    """
    def __init__(self, history_days=2 * 366, features_cols=[], **prophet_kwargs):
        """
        :param history_days: Nombre de jours d'historique  conserver pour l'entranement.
        :param features_cols: Liste des colonnes  utiliser comme features supplmentaires (ex: temprature).
        :param prophet_kwargs: Autres arguments  passer  Prophet (ex: yearly_seasonality=True)
        """
        self.history_days = history_days
        self.features_cols = features_cols

        default_kwargs = {
            'daily_seasonality': False,
            'yearly_seasonality': 'auto',
            'weekly_seasonality': 'auto'
        }
        default_kwargs.update(prophet_kwargs)
        self.prophet_kwargs = default_kwargs
        self.model = Prophet()


    def _prepare_df(self, df_source, is_fit=False):
        df_p = pd.DataFrame({
            'ds': df_source['cible_timestamp_paris'].dt.tz_localize(None),
            'y': df_source.get('cible_consommation_mw', np.nan)
        })

        X_feat = df_source[self.features_cols].copy()
        cat_cols = X_feat.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            X_feat = pd.get_dummies(X_feat, columns=cat_cols, drop_first=False, dtype=float)

        if is_fit:
            self.encoded_cols = X_feat.columns.tolist()
        else:
            for col in getattr(self, 'encoded_cols', []):
                if col not in X_feat.columns:
                    X_feat[col] = 0.0
            X_feat = X_feat[getattr(self, 'encoded_cols', X_feat.columns.tolist())]

        for col in X_feat.columns:
            df_p[col] = X_feat[col].values

        is_weekend = df_p['ds'].dt.dayofweek >= 5
        is_weekday = ~is_weekend

        df_p['weekday'] = is_weekday
        df_p['weekend'] = is_weekend

        return df_p


    def fit(self, df_train):
        if self.history_days is not None:
            cutoff_date = df_train['cible_timestamp_paris'].max() - pd.DateOffset(days=self.history_days)
            df_fit = df_train[df_train['cible_timestamp_paris'] >= cutoff_date].copy()
        else:
            df_fit = df_train.copy()

        df_prophet = self._prepare_df(df_fit, is_fit=True)
        
        # On dropna dynamiquement sur les colonnes encodes
        df_prophet = df_prophet.dropna(subset=['y'] + self.encoded_cols)

        self.model = Prophet(**self.prophet_kwargs)
        self.model.add_country_holidays(country_name='FR')

        self.model.add_seasonality(name='daily_wd', period=1, fourier_order=10, condition_name='weekday')
        self.model.add_seasonality(name='daily_we', period=1, fourier_order=10, condition_name='weekend')

        for feature in self.encoded_cols:
            self.model.add_regressor(feature)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            self.model.fit(df_prophet)


    def predict(self, df_test):
        df_future = self._prepare_df(df_test, is_fit=False)
        forecast = self.model.predict(df_future)
        return pd.Series(forecast['yhat'].values, index=df_test.index)


    def get_feature_importances(self):
        if self.model is None or not self.features_cols:
            return None

        from prophet.utilities import regressor_coefficients
        try:
            df_imp = regressor_coefficients(self.model)
            df_imp = df_imp.rename(columns={'regressor': 'Feature', 'coef': 'Importance'})
            df_imp['Abs_Importance'] = df_imp['Importance'].abs()
            return df_imp.sort_values(by='Abs_Importance', ascending=False)[['Feature', 'Importance']]
        except Exception:
            return None
