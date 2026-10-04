import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class TimeSeriesEvaluator:
    """
    Classe universelle pour évaluer les modèles de prévision temporelle.
    Gère le découpage chronologique, l'entraînement, l'inférence et le calcul des métriques.
    """
    def __init__(self, df, target_col='consommation_mw', date_col='timestamp_paris'):
        self.df = df.sort_values(date_col).copy()
        self.target_col = target_col
        self.date_col = date_col
        self.results = {}
        self.predictions = {}

    def split_data(self, val_start, test_start):
        train = self.df[self.df[self.date_col] < val_start].copy()
        val = self.df[(self.df[self.date_col] >= val_start) & (self.df[self.date_col] < test_start)].copy()
        test = self.df[self.df[self.date_col] >= test_start].copy()

        print(f"--- Découpage Temporel ---")
        print(
            f"Train: {len(train)} lignes ({len(train) / len(self.df) * 100:.1f}%) |\n"
            f"Val: {len(val)} lignes ({len(val) / len(self.df) * 100:.1f}%) |\n"
            f"Test: {len(test)} lignes ({len(test) / len(self.df) * 100:.1f}%)\n"
        )
        return train, val, test

    def evaluate_model(self, model, df_train, df_test):
        """
        Entraîne le modèle sur le Train et calcule les métriques sur le Test.
        """
        model_name = model.__class__.__name__

        print(f"[{model_name}] Entraînement en cours...")
        model.fit(df_train)

        print(f"[{model_name}] Prédiction en cours...")
        preds = model.predict(df_test)

        y_test = df_test[self.target_col]

        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        mape = (abs((y_test - preds) / y_test).mean()) * 100

        print(f"RÉSULTATS {model_name} : MAE = {mae:.2f} MW | RMSE = {rmse:.2f} MW | MAPE = {mape:.2f}%\n")

        self.results[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        self.predictions[model_name] = pd.Series(preds, index=df_test.index)

        return preds

    def rolling_validation(self, model, start_date, end_date, step_size='1D'):
        """
        Validation walk-forward unifiée.
        Permet de tester avec n'importe quelle fréquence de réentraînement.
        - step_size='1D' : ré-entraînement quotidien
        - step_size='1MS' : ré-entraînement mensuel
        - step_size='1YS' : ré-entraînement annuel
        """
        model_name = model.__class__.__name__

        start = pd.to_datetime(start_date).tz_localize('Europe/Paris')
        end = pd.to_datetime(end_date).tz_localize('Europe/Paris')

        # Génération des dates de coupure (cutoffs)
        cutoffs = pd.date_range(start=start, end=end, freq=step_size)
        if len(cutoffs) == 0 or cutoffs[-1] < end:
            cutoffs = cutoffs.append(pd.DatetimeIndex([end]))

        print(f"\n========== ROLLING VALIDATION : {model_name} ==========")
        print(f"Période: {start_date} -> {end_date} | Pas de réentraînement: {step_size} | {len(cutoffs)-1} itérations")

        all_preds = []
        all_y = []

        for i in range(len(cutoffs) - 1):
            current_date = cutoffs[i]
            next_date = cutoffs[i+1]

            df_train = self.df[self.df[self.date_col] < current_date].copy()
            df_test = self.df[(self.df[self.date_col] >= current_date) & (self.df[self.date_col] < next_date)].copy()

            if not df_test.empty:
                model.fit(df_train)
                preds = model.predict(df_test)

                all_preds.extend(preds)
                all_y.extend(df_test[self.target_col].values)

            # Affichage adaptatif pour ne pas spammer la console si itérations quotidiennes
            if len(cutoffs) > 20:
                sys.stdout.write(f"\rProgression: {current_date.strftime('%Y-%m-%d')}...")
                sys.stdout.flush()
            else:
                if not df_test.empty:
                    mae = mean_absolute_error(df_test[self.target_col], preds)
                    print(f"[{current_date.strftime('%Y-%m-%d')}] MAE: {mae:.2f} MW")

        mae = mean_absolute_error(all_y, all_preds)
        rmse = root_mean_squared_error(all_y, all_preds)
        mape = (np.abs((np.array(all_y) - np.array(all_preds)) / np.array(all_y)).mean()) * 100

        print(f"\n\nSCORE GLOBAL ROLLING ({step_size}) : MAE = {mae:.2f} MW | RMSE = {rmse:.2f} MW | MAPE = {mape:.2f}%\n")

        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    def plot_predictions(self, df_test, start_date=None, end_date=None):
        plt.figure(figsize=(15, 6))

        mask = pd.Series(True, index=df_test.index)
        if start_date:
            mask = mask & (df_test[self.date_col] >= start_date)
        if end_date:
            mask = mask & (df_test[self.date_col] <= end_date)

        df_plot = df_test[mask]

        plt.plot(df_plot[self.date_col], df_plot[self.target_col], label='Vraie Consommation', color='black', linewidth=2)

        colors = ['#D90429', '#2E86AB', '#F5A623', '#82C0CC']
        for i, (name, preds) in enumerate(self.predictions.items()):
            if name in self.results: # Only plot models from standard evaluate
                pred_plot = preds[mask]
                plt.plot(df_plot[self.date_col], pred_plot, label=f'Préd: {name}', alpha=0.8, color=colors[i % len(colors)])

        plt.title("Comparaison des Modèles : Prédictions vs Réel")
        plt.ylabel("Consommation (MW)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
