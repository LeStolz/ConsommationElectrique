from .utils import Regressor
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class FourierRegressor(Regressor):
    """
    Régression linéaire enrichie de bases de Fourier pour modéliser
    les multiples saisonnalités (journalière, hebdomadaire, annuelle)
    avec différents ordres d'harmoniques.
    """
    def __init__(self, model=LinearRegression, features_cols=[], fourier_orders={'yearly': 5, 'weekly': 3, 'daily': 5}):
        self.model = model()
        self.features_cols = features_cols
        self.fourier_orders = fourier_orders
        self.encoded_cols = []

    def _add_fourier_terms(self, X, df):
        ts = pd.to_datetime(df['cible_timestamp_paris'])

        # Yearly seasonality (periode = 365.25 jours)
        if 'yearly' in self.fourier_orders:
            day_of_year = ts.dt.dayofyear + (ts.dt.hour / 24.0)
            for k in range(1, self.fourier_orders['yearly'] + 1):
                X[f'fourier_yearly_sin_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
                X[f'fourier_yearly_cos_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)

        # Weekly seasonality (periode = 7 jours)
        if 'weekly' in self.fourier_orders:
            day_of_week = ts.dt.dayofweek + (ts.dt.hour / 24.0)
            for k in range(1, self.fourier_orders['weekly'] + 1):
                X[f'fourier_weekly_sin_{k}'] = np.sin(2 * np.pi * k * day_of_week / 7.0)
                X[f'fourier_weekly_cos_{k}'] = np.cos(2 * np.pi * k * day_of_week / 7.0)

        # Daily seasonality (periode = 24 heures)
        if 'daily' in self.fourier_orders:
            hour = ts.dt.hour + (ts.dt.minute / 60.0)
            for k in range(1, self.fourier_orders['daily'] + 1):
                X[f'fourier_daily_sin_{k}'] = np.sin(2 * np.pi * k * hour / 24.0)
                X[f'fourier_daily_cos_{k}'] = np.cos(2 * np.pi * k * hour / 24.0)

        return X

    def _prepare_data(self, df, is_fit=False):
        X = df[self.features_cols].copy()

        X = self._add_fourier_terms(X, df)

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

        df_imp = pd.DataFrame({
            'Feature': self.encoded_cols,
            'Importance': self.model.coef_
        })
        df_imp['Abs_Importance'] = df_imp['Importance'].abs()
        return df_imp.sort_values(by='Abs_Importance', ascending=False)[['Feature', 'Importance']]