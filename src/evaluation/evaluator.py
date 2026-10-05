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
    def __init__(self, df, target_col='consommation_mw', date_col='timestamp_utc'):
        self.df = df.sort_values(date_col).copy()
        self.target_col = target_col
        self.date_col = date_col

        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], utc=True)

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

    def rolling_validation(self, model, df_train, df_val, step_size_days=1, verbose=True):
        """
        Validation walk-forward unifiée.
        - df_train : Historique initial
        - df_val : Données sur lesquelles on va évaluer de manière glissante
        - step_size_hours : pas de réentraînement en jours (ex: 1 pour quotidien, 7 pour hebdomadaire)
        """
        import numpy as np
        import sys

        model_name = model.__class__.__name__

        df_combined = pd.concat([df_train, df_val]).sort_values(self.date_col)
        start = df_val[self.date_col].min()
        end = df_val[self.date_col].max() + pd.Timedelta(hours=1)

        # Génération des dates de coupure (cutoffs) avec l'intervalle en heures
        cutoffs = pd.date_range(start=start, end=end, freq=f'{step_size_days}D')
        if len(cutoffs) == 0 or cutoffs[-1] < end:
            cutoffs = cutoffs.append(pd.DatetimeIndex([end]))

        if verbose:
            print(f"\n========== ROLLING VALIDATION : {model_name} ==========")
        if verbose:
            print(f"Évaluation sur {len(df_val)} lignes | Pas de réentraînement: {step_size_days} jours | {len(cutoffs)-1} itérations")

        all_preds = []
        all_y = []

        for i in range(len(cutoffs) - 1):
            current_date = cutoffs[i]
            next_date = cutoffs[i+1]

            # Le Train grossit en absorbant progressivement df_val
            current_train = df_combined[df_combined[self.date_col] < current_date].copy()
            current_val = df_combined[(df_combined[self.date_col] >= current_date) & (df_combined[self.date_col] < next_date)].copy()

            # --- SÉCURITÉ ANTI DATA-LEAK ---
            # Au lieu de supprimer physiquement les lignes (ce qui détruirait le décalage de `.shift(24)`),
            # on censure (remplace par NaN) toutes les données du jour J après 14h.
            if not current_train.empty:
                last_date = current_train[self.date_col].dt.date.max()
                mask_leak = (current_train[self.date_col].dt.date == last_date) & (current_train[self.date_col].dt.hour > 14)

                # On ne masque que les colonnes de type float (consommation, température, vent, etc.)
                # Les variables booléennes ou entières (calendrier, férié) restent connues à 14h !
                float_cols = current_train.select_dtypes(include=['float', 'float32', 'float64']).columns
                cols_to_mask = [c for c in float_cols if c not in ['timestamp_utc', 'timestamp_paris']]
                current_train.loc[mask_leak, cols_to_mask] = np.nan

            if not current_val.empty:
                model.fit(current_train)

                # Censure totale du Test Set (J+1) avant la prédiction
                # On masque toutes les valeurs physiques (floats) du futur
                # (Consommation, Température réelle) pour empêcher toute triche !
                current_val_pred = current_val.copy()
                current_val_pred[cols_to_mask] = np.nan

                preds = model.predict(current_val_pred)

                all_preds.extend(preds)
                all_y.extend(current_val[self.target_col].values)

            # Affichage adaptatif
            if len(cutoffs) > 20:
                if verbose:
                    sys.stdout.write(f"\rProgression: {current_date.strftime('%Y-%m-%d %H:%M')}...")
                    sys.stdout.flush()
            else:
                if not current_val.empty:
                    mae = mean_absolute_error(current_val[self.target_col], preds)
                    if verbose:
                        print(f"[{current_date.strftime('%Y-%m-%d')}] MAE: {mae:.2f} MW")

        mae = mean_absolute_error(all_y, all_preds)
        rmse = root_mean_squared_error(all_y, all_preds)
        mape = (np.abs((np.array(all_y) - np.array(all_preds)) / np.array(all_y)).mean()) * 100

        if verbose:
            print(f"\n\nSCORE GLOBAL ROLLING ({step_size_days} jours) : MAE = {mae:.2f} MW | RMSE = {rmse:.2f} MW | MAPE = {mape:.2f}%\n")

        self.results[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        self.predictions[model_name] = pd.Series(all_preds, index=df_val.index)

        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


    def grid_search_rolling_validation(self, model_class, param_grid, df_train, df_val, step_size_days=1, metric='MAE'):
        """
        Recherche par grille des meilleurs hyperparamètres avec validation glissante.
        - model_class : La classe du modèle (ex: LinearRegressor)
        - param_grid : Dictionnaire des paramètres à tester (ex: {'n_weeks': [2, 3, 4]})
        """
        from sklearn.model_selection import ParameterGrid
        import time

        best_score = float('inf')
        best_params = None
        best_model_name = model_class.__name__
        results = []

        grid = list(ParameterGrid(param_grid))
        print(f"\n========== GRID SEARCH : {best_model_name} ==========")
        print(f"Total combinations to evaluate: {len(grid)}\n")

        for i, params in enumerate(grid):
            print(f"--- Combination {i+1}/{len(grid)} : {params} ---")
            start_time = time.time()

            # Instanciate the model
            model = model_class(**params)

            # Run rolling validation silently
            score_dict = self.rolling_validation(model, df_train, df_val, step_size_days=step_size_days, verbose=False)
            score = score_dict[metric]

            elapsed = time.time() - start_time
            print(f"    -> {metric}: {score:.2f} MW (took {elapsed:.1f}s)")

            results.append({'params': params, 'score': score, 'details': score_dict})

            if score < best_score:
                best_score = score
                best_params = params

        print(f"\n========== GRID SEARCH RESULTS ==========")
        print(f"Best {metric} : {best_score:.2f} MW")
        print(f"Best Params : {best_params}\n")

        return best_params, best_score, results


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