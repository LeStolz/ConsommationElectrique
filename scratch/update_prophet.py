import re

filepath = r'e:\Projects\Studies\ElectricityConsumption\src\models\prophet.py'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

replacement_prepare = '''    def _prepare_df(self, df_source):
        df_p = pd.DataFrame({
            'ds': df_source['cible_timestamp_paris'].dt.tz_localize(None),
            'y': df_source.get('cible_consommation_mw', np.nan)
        })

        for feature in self.features_cols:
            df_p[feature] = df_source[feature].values

        is_weekend = df_p['ds'].dt.dayofweek >= 5
        is_weekday = ~is_weekend
        
        # En France, l'hiver (novembre à mars) a une pointe très marquée à 19h
        is_winter = df_p['ds'].dt.month.isin([11, 12, 1, 2, 3])
        is_summer = ~is_winter

        df_p['winter_weekday'] = is_winter & is_weekday
        df_p['winter_weekend'] = is_winter & is_weekend
        df_p['summer_weekday'] = is_summer & is_weekday
        df_p['summer_weekend'] = is_summer & is_weekend

        return df_p'''

replacement_fit = '''    def fit(self, df_train):
        if self.history_days is not None:
            cutoff_date = df_train['cible_timestamp_paris'].max() - pd.DateOffset(days=self.history_days)
            df_fit = df_train[df_train['cible_timestamp_paris'] >= cutoff_date].copy()
        else:
            df_fit = df_train.copy()

        df_prophet = self._prepare_df(df_fit)
        df_prophet = df_prophet.dropna(subset=['y'] + self.features_cols)

        self.model = Prophet(**self.prophet_kwargs)
        self.model.add_country_holidays(country_name='FR')

        # 4 Profils journaliers avec fourier_order=15 pour capter la pointe de 19h !
        self.model.add_seasonality(name='daily_winter_wd', period=1, fourier_order=15, condition_name='winter_weekday')
        self.model.add_seasonality(name='daily_winter_we', period=1, fourier_order=15, condition_name='winter_weekend')
        self.model.add_seasonality(name='daily_summer_wd', period=1, fourier_order=15, condition_name='summer_weekday')
        self.model.add_seasonality(name='daily_summer_we', period=1, fourier_order=15, condition_name='summer_weekend')

        for feature in self.features_cols:
            self.model.add_regressor(feature)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            self.model.fit(df_prophet)'''

text = re.sub(r'    def _prepare_df.*?return df_p', replacement_prepare, text, flags=re.DOTALL)
text = re.sub(r'    def fit\(self, df_train\):.*?self\.model\.fit\(df_prophet\)', replacement_fit, text, flags=re.DOTALL)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated prophet.py!")
