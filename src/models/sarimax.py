import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXRegressor:
    def __init__(self, features_cols, order, seasonal_order, since_year):
        """
        since_year: SARIMAX est très lent. Il est conseillé de limiter
        l'historique (ex: since_year=2023) plutôt que d'utiliser des années entières.
        """
        self.features_cols = features_cols if features_cols else []
        self.order = order
        self.seasonal_order = seasonal_order
        self.since_year = since_year
        self.model_res = None

        self.date_col = 'cible_timestamp_paris'
        self.target_col = 'cible_consommation_mw'


    def fit(self, df_train):
        self.history = df_train.sort_values(self.date_col).copy()

        if self.since_year is not None:
            self.history = self.history[self.history[self.date_col].dt.year >= self.since_year]

        # On garde une trace de la fin du train pour s'assurer que le predict enchaîne
        self.last_train_date = self.history[self.date_col].max()

        y = self.history[self.target_col].values

        exog = None
        if self.features_cols:
            exog = self.history[self.features_cols].astype(float).values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_res = model.fit(disp=False)
            
        # Stocker les prédictions in-sample pour l'évaluation sur le train_set
        self.fitted_series = pd.Series(self.model_res.fittedvalues, index=self.history.index)

    def predict(self, df_test):
        df_pred = df_test.sort_values(self.date_col)
        
        # Intercepter l'évaluation in-sample (train_eval)
        is_in_sample = df_pred[self.date_col].max() <= self.last_train_date
        if is_in_sample:
            mapped = df_pred.index.map(self.fitted_series)
            return pd.Series(mapped, index=df_pred.index).fillna(0)
            
        gap = getattr(self, 'gap_data', pd.DataFrame())
        
        res = self.model_res

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if not gap.empty:
                gap = gap.sort_values(self.date_col)
                
                # Les 14 premieres heures sont connues, on met a jour le filtre de Kalman !
                known_mask = gap[self.date_col].dt.hour < 14
                gap_known = gap[known_mask]
                gap_unknown = gap[~known_mask]
                
                if not gap_known.empty:
                    y_known = gap_known[self.target_col].values
                    exog_known = gap_known[self.features_cols].astype(float).values if self.features_cols else None
                    # append() met a jour la memoire SANS re-entrainer (refit=False)
                    res = res.append(endog=y_known, exog=exog_known, refit=False)
                
                # On forecast les heures inconnues du gap + le test set
                df_combined_future = pd.concat([gap_unknown, df_pred])
                exog_future = df_combined_future[self.features_cols].astype(float).values if self.features_cols else None
                
                forecast = res.forecast(steps=len(df_combined_future), exog=exog_future)
                
                # On jette les heures du gap pour ne renvoyer que df_test
                if len(gap_unknown) > 0:
                    forecast = forecast[len(gap_unknown):]
                    
                return pd.Series(forecast, index=df_pred.index)
            else:
                exog = df_pred[self.features_cols].astype(float).values if self.features_cols else None
                forecast = res.forecast(steps=len(df_pred), exog=exog)
                return pd.Series(forecast, index=df_pred.index)
