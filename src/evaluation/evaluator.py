import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import ParameterGrid
import time, sys
import numpy as np

class TimeSeriesEvaluator:
    """
    Classe universelle pour évaluer les modèles de prévision temporelle.
    Gère le découpage chronologique, l'entraînement, l'inférence et le calcul des métriques.
    """
    def __init__(self, df, target_col='consommation_mw', pred_hour_local=14):
        self.df = df.sort_values('timestamp_paris').copy()
        self.target_col = target_col
        self.date_col = 'timestamp_paris'
        self.pred_hour_local = pred_hour_local

        self.df['timestamp_utc'] = pd.to_datetime(self.df['timestamp_utc'], utc=True)
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], utc=True).dt.tz_convert('Europe/Paris')

        self.results = {}
        self.predictions = {}


    def split_data(self, val_start, test_start):
        train = self.df[self.df[self.date_col] < val_start].copy()
        val = self.df[(self.df[self.date_col] >= val_start) & (self.df[self.date_col] < test_start)].copy()
        test = self.df[self.df[self.date_col] >= test_start].copy()

        print(f"Découpage Temporel")
        print(
            f"Train: {len(train)} lignes ({len(train) / len(self.df) * 100:.1f}%)\n"
            f"Val: {len(val)} lignes ({len(val) / len(self.df) * 100:.1f}%)\n"
            f"Test: {len(test)} lignes ({len(test) / len(self.df) * 100:.1f}%)\n"
        )
        return train, val, test


    def rolling_validation(self, model, df_train, df_val, freq="1D"):
        """
        Validation walk-forward unifiée.
        - df_train : Historique initial
        - df_val : Données sur lesquelles on va évaluer de manière glissante
        - freq : Fréquence de réentraînement (ex: "1D" pour quotidien, "7D" pour hebdomadaire)
        """
        model_name = model.__class__.__name__
        df_combined = pd.concat([df_train, df_val]).sort_values(self.date_col)

        start = df_val[self.date_col].dt.normalize().min()
        end = df_val[self.date_col].dt.normalize().max() + pd.Timedelta(days=1)

        # Génération des dates de coupure (cutoffs)
        cutoffs = pd.date_range(start=start, end=end, freq=freq)
        if len(cutoffs) == 0 or cutoffs[-1] < end:
            cutoffs = cutoffs.append(pd.DatetimeIndex([end]))

        print(f"Évaluation de {model_name} | {len(df_val)} lignes | réentraînement/{freq} | {len(cutoffs)-1} itérations")

        all_preds = []
        all_y = []

        for i in range(len(cutoffs) - 1):
            current_date = cutoffs[i]
            next_date = cutoffs[i + 1]

            current_train = df_combined[df_combined[self.date_col] < current_date].copy()
            current_val = df_combined[(df_combined[self.date_col] >= current_date) & (df_combined[self.date_col] < next_date)].copy()

            # On censure toutes les données du jour J après 14h.
            last_date = current_train[self.date_col].dt.date.max()
            mask_leak = \
                (current_train[self.date_col].dt.date == last_date) & \
                (current_train[self.date_col].dt.hour >= self.pred_hour_local)

            float_cols = current_train.select_dtypes(include=['float', 'float32', 'float64']).columns
            cols_to_mask = [c for c in float_cols if c not in ['timestamp_utc', 'timestamp_paris']]
            current_train.loc[mask_leak, cols_to_mask] = np.nan

            if not current_val.empty:
                model.fit(current_train)

                current_val_pred = current_val.copy()
                current_val_pred[cols_to_mask] = np.nan

                preds = model.predict(current_val_pred)

                all_preds.extend(preds)
                all_y.extend(current_val[self.target_col].values)

                sys.stdout.write(f"\rProgression: {current_date.strftime('%Y-%m-%d %H:%M')}...")
                sys.stdout.flush()

        mae = mean_absolute_error(all_y, all_preds)
        rmse = root_mean_squared_error(all_y, all_preds)
        mape = (np.abs((np.array(all_y) - np.array(all_preds)) / np.array(all_y)).mean()) * 100

        print(f"\n\nScore : MAE = {mae:.2f} MW | RMSE = {rmse:.2f} MW | MAPE = {mape:.2f}%\n")

        self.results[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        self.predictions[model_name] = pd.Series(all_preds, index=df_val.index)

        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


    def grid_search_rolling_validation(self, model_class, param_grid, df_train, df_val, freq="1D", metric='MAE'):
        """
        Recherche par grille des meilleurs hyperparamètres avec validation glissante.
        """
        best_score = float('inf')
        best_params = None
        best_model_name = model_class.__name__
        results = []

        grid = list(ParameterGrid(param_grid))
        print(f"Grid Search de {best_model_name} avec {len(grid)} combinaisons de paramètres.\n")

        for i, params in enumerate(grid):
            print(f"Combination {i+1}/{len(grid)} : {params}")
            start_time = time.time()

            model = model_class(**params)

            score_dict = self.rolling_validation(model, df_train, df_val, freq=freq)
            score = score_dict[metric]

            elapsed = time.time() - start_time
            print(f"-> {metric}: {score:.2f} MW (took {elapsed:.1f}s)")

            results.append({'params': params, 'score': score, 'details': score_dict})

            if score < best_score:
                best_score = score
                best_params = params

        print(f"Best {metric} : {best_score:.2f} MW")
        print(f"Best Params : {best_params}\n")

        return best_params, best_score, results


    def plot_predictions(self, df_test, start_date=None, end_date=None):
        mask = pd.Series(True, index=df_test.index)
        if start_date:
            mask = mask & (df_test[self.date_col] >= start_date)
        if end_date:
            mask = mask & (df_test[self.date_col] <= end_date)

        df_plot = df_test[mask]

        models = [
            name for name in self.predictions
            if name in self.results
        ]

        n_models = len(models)

        fig, axes = plt.subplots(
            n_models,
            1,
            figsize=(16, 5 * n_models),
            sharex=True
        )

        for ax, name in zip(axes, models):
            preds = self.predictions[name]
            pred_plot = preds[mask]

            # Real consumption
            ax.plot(
                df_plot[self.date_col],
                df_plot[self.target_col],
                label="Vraie Consommation",
                color="black",
                linewidth=2
            )

            # Model prediction
            ax.plot(
                df_plot[self.date_col],
                pred_plot,
                label=name,
                alpha=0.8
            )

            ax.set_title(name)
            ax.set_ylabel("Consommation (MW)")
            ax.legend()
            ax.grid(True, alpha=0.3)


            axes[-1].set_xlabel("Date")

        fig.suptitle("Comparaison des Modèles : Prédictions vs Réel", fontsize=14)
        plt.ylabel("Consommation (MW)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()