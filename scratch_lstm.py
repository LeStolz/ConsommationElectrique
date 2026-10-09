with open('src/models/lstm_regressor.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx1 = text.find('def fit(self, df_train: pd.DataFrame):')

if idx1 != -1:
    replacement = '''def fit(self, df_train: pd.DataFrame):
        # Save df_train as our raw historical data (no need to read CSV)
        self.df_raw = df_train.copy()
        
        # Ensure datetimes are correct
        if not pd.api.types.is_datetime64_any_dtype(self.df_raw['timestamp_paris']):
            self.df_raw['timestamp_paris'] = pd.to_datetime(self.df_raw['timestamp_paris'], utc=True).dt.tz_convert('Europe/Paris')
        if not pd.api.types.is_datetime64_any_dtype(self.df_raw['timestamp_utc']):
            self.df_raw['timestamp_utc'] = pd.to_datetime(self.df_raw['timestamp_utc'], utc=True)

        # Find the unique prevision dates from the evaluator's df_train
        jours_train = pd.to_datetime(df_train['prevision_date']).dt.normalize().unique()

        # We need a validation split for early stopping.
        split_idx = int(len(jours_train) * 0.9)
        jours_t = jours_train[:split_idx]
        jours_v = jours_train[split_idx:]

        # Build datasets
        jeu_train = construire_jeu(self.df_raw, jours=jours_t, avec_cible=True)
        self.scalers = jeu_train[4] # Save scalers

        jeu_val = construire_jeu(self.df_raw, scalers=self.scalers, jours=jours_v, avec_cible=True)

        self.models = entrainer_ensemble(jeu_train, jeu_val, n_modeles=self.n_modeles, max_epochs=self.max_epochs, **self.kwargs)

    def predict(self, df_test: pd.DataFrame):
        jours_test = pd.to_datetime(df_test['prevision_date']).dt.normalize().unique()
        
        # Merge history, gap (Day J) and df_test (Day J+1) so rolling features compute properly
        gap = getattr(self, 'gap_data', pd.DataFrame())
        df_combined = pd.concat([self.df_raw, gap, df_test])
        
        if not pd.api.types.is_datetime64_any_dtype(df_combined['timestamp_paris']):
            df_combined['timestamp_paris'] = pd.to_datetime(df_combined['timestamp_paris'], utc=True).dt.tz_convert('Europe/Paris')
        if not pd.api.types.is_datetime64_any_dtype(df_combined['timestamp_utc']):
            df_combined['timestamp_utc'] = pd.to_datetime(df_combined['timestamp_utc'], utc=True)
            
        jeu_test = construire_jeu(df_combined, scalers=self.scalers, jours=jours_test, avec_cible=False)

        preds_24h = predire(self.models, jeu_test, self.scalers) # Shape: (N_jours, 24)

        pred_dict = {}
        dates_J = jeu_test[3]
        for i, date_j in enumerate(dates_J):
            pred_dict[date_j] = preds_24h[i]

        final_prediction = pd.Series(index=df_test.index, dtype=float)
        for idx, row in df_test.iterrows():
            d = pd.to_datetime(row['prevision_date']).normalize()
            h = pd.to_datetime(row['cible_timestamp_paris']).hour
            if d in pred_dict:
                final_prediction.loc[idx] = pred_dict[d][h]
            else:
                final_prediction.loc[idx] = row['consommation_mw_meme_heure_derniere_connue']

        return final_prediction
'''
    text = text[:idx1] + replacement
    with open('src/models/lstm_regressor.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Done!')
