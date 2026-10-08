import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import ParameterGrid
from pathlib import Path

project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)


from src.models.utils import get_lagged_local_value, get_latest_local_value


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


COLONNES_METEO = [
    "temperature_c_pondere_pop",
    # "temperature_c",
    # "temperature_point_rosee_c",
    # "temperature_point_rosee_c_pondere_pop",
    # "humidite_pct", "humidite_pct_pondere_pop",
    # "vent_direction_deg", "vent_direction_deg_pondere_pop",
    # "vent_vitesse_ms", "vent_vitesse_ms_pondere_pop",
    # "nebulosite", "nebulosite_pondere_pop",
    # "precip_1h_mm", "precip_1h_mm_pondere_pop",
]

COLONNES_CALENDRIER = [
    "heure", "jour_semaine", "jour_annee",
    "mois", "saison", "saison_meteorologique", "annee", "weekend", "ferie", "vacances",
    "confinement_numero",
]


class TimeSeriesEvaluator:
    """
    Classe universelle pour évaluer les modèles de prévision temporelle.
    Gère le découpage chronologique, l'entraînement, l'inférence et le calcul des métriques.
    """
    def __init__(self, df, scenario, pred_hour_local=14, features_cols=None):
        if features_cols is None:
            features_cols = [
                "consommation_mw_moins_7",
                "consommation_mw_moins_365",
                "consommation_mw_moins_366",
                "consommation_mw_meme_heure_derniere_connue",
                "consommation_mw_derniere_connue",
                "derive_consommation_mw"
            ] + [
                feature
                for col in COLONNES_METEO
                for feature in [
                    f"{col}_moins_365",
                    f"{col}_moins_366",
                    f"{col}_meme_heure_derniere_connue",
                    f"{col}_derniere_connue",
                    f"derive_{col}"
                ]
            ]

        self.df = df.sort_values('timestamp_paris').copy()
        self.target_col = 'cible_consommation_mw'
        self.date_col = 'cible_timestamp_paris'
        self.pred_hour_local = pred_hour_local
        self.scenario = scenario
        self.features_cols = features_cols

        self.df['timestamp_utc'] = pd.to_datetime(self.df['timestamp_utc'], utc=True)
        self.df['timestamp_paris'] = pd.to_datetime(self.df['timestamp_paris'], utc=True).dt.tz_convert('Europe/Paris')

        self.predictions = {}
        self.models = {}

        self.df_train = None
        self.df_val = None
        self.df_test = None


    def _normale_saisonniere(self, df):
        """Température pondérée 'normale' de chaque jour de l'année, apprise
        uniquement sur les années <= 2023 pour éviter la fuite val/test."""
        if "temperature_c_pondere_pop" not in df.columns or "jour_annee" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        t = df["temperature_c_pondere_pop"]
        masque = df["timestamp_paris"].dt.year <= 2023
        if not masque.any():
            return pd.Series(np.nan, index=df.index)

        moy = t[masque].groupby(df.loc[masque, "jour_annee"]).mean().reindex(range(1, 367))
        moy = moy.interpolate(limit_direction="both")
        ext = pd.concat([moy.iloc[-15:], moy, moy.iloc[:15]])
        lisse = ext.rolling(31, center=True, min_periods=1).mean().iloc[15:-15]
        lisse.index = moy.index
        return df["jour_annee"].map(lisse)


    def _construct_useful_features(self, source_df):
        df = source_df.copy()
        df['date'] = df['timestamp_paris'].dt.date

        # --- météo : on génère systématiquement les deux versions ---
        # "parfait_* " = observée à l'heure visée (fuite assumée)
        # "*" = dernière valeur connue à 14h le jour J
        df['saison_meteorologique'] = df['timestamp_paris'].dt.month.map({
            12: "Hiver", 1: "Hiver", 2: "Hiver",
            3: "Printemps", 4: "Printemps", 5: "Printemps",
            6: "Été", 7: "Été", 8: "Été",
            9: "Automne", 10: "Automne", 11: "Automne"
        })

        for col in self.features_cols.copy():
            if not col.startswith("derive_"):
                continue

            base_col = col.replace("derive_", "")
            t = df[base_col]
            df[f"{base_col}_moyenne_24h"] = t.rolling(24, min_periods=12).mean()
            df[f"{base_col}_min_24h"] = t.rolling(24, min_periods=12).min()
            df[f"{base_col}_max_24h"] = t.rolling(24, min_periods=12).max()
            df[f"{base_col}_moyenne_48h"] = t.rolling(window=48, min_periods=24).mean()
            df[f"{base_col}_moyenne_72h"] = t.rolling(window=72, min_periods=36).mean()
            df[f"{base_col}_tendance_6h"] = t - t.shift(6)
            df[f"{base_col}_tendance_24h"] = t - t.shift(24)

            new_features = [
                f"{base_col}_moyenne_24h_derniere_connue",
                f"{base_col}_min_24h_derniere_connue",
                f"{base_col}_max_24h_derniere_connue",
                f"{base_col}_moyenne_48h_derniere_connue",
                f"{base_col}_moyenne_72h_derniere_connue",
                f"{base_col}_tendance_6h_derniere_connue",
                f"{base_col}_tendance_24h_derniere_connue",
            ]
            self.features_cols.extend(new_features)
            self.features_cols.remove(col)

        col_temp = "temperature_c_pondere_pop"
        df[f"{col_temp}_sous_15"] = (15 - df[col_temp]).clip(lower=0)
        df[f"{col_temp}_au_dessus_22"] = (df[col_temp] - 22).clip(lower=0)
        f = (15 - df[col_temp]).clip(lower=0)
        df[f"{col_temp}_sous_15_cumul_3j"] = f.rolling(72, min_periods=36).mean()

        df[f"{col_temp}_ecart_normale"] = \
            df[f"{col_temp}_moyenne_24h"] - self._normale_saisonniere(df)

        self.features_cols.extend([
            f"{col_temp}_sous_15_derniere_connue",
            f"{col_temp}_au_dessus_22_derniere_connue",
            f"{col_temp}_sous_15_cumul_3j_derniere_connue",
            f"{col_temp}_ecart_normale_derniere_connue"
        ])

        last_cols = [
            (col.split('_moins_')[0], int(col.split('_moins_')[1]))
            for col in self.features_cols if '_moins_' in col
        ]

        lastest_known_cols = [
            col.split('_derniere_connue')[0]
            for col in self.features_cols if '_derniere_connue' in col and '_meme_heure' not in col
        ]

        lastest_known_same_hour_cols = [
            col.split('_meme_heure_derniere_connue')[0]
            for col in self.features_cols if '_meme_heure_derniere_connue' in col
        ]

        for col, lag in last_cols:
            df[f'{col}_moins_{lag}'] = get_lagged_local_value(df, df, lag_days=lag, col=col)

        cutoff = df.loc[
            df['timestamp_paris'].dt.hour == self.pred_hour_local - 1, df.columns
        ].copy()
        cutoff['date'] = (pd.to_datetime(cutoff['date']) + pd.DateOffset(days=1)).dt.date

        for col in lastest_known_cols:
            if col in cutoff.columns:
                if self.scenario == "perfect":
                    # Pour "parfait", on triche et on utilise la valeur à l'heure cible (fuite volontaire)
                    df[f'{col}_derniere_connue'] = df[col]
                else:
                    # Pour réaliste, on utilise la valeur coupée à 14h
                    col_cutoff = cutoff.set_index('date')[col]
                    df[f'{col}_derniere_connue'] = df['date'].map(col_cutoff)

        for col in lastest_known_same_hour_cols:
            if self.scenario == "perfect":
                df[f'{col}_meme_heure_derniere_connue'] = get_lagged_local_value(df, df, lag_days=1, col=col)
            else:
                df[f'{col}_meme_heure_derniere_connue'] = get_latest_local_value(df, df, col, self.pred_hour_local)

        return df


    def construct_features(self, source_df):
        """Construit la table d'entraînement/évaluation :
        une ligne par (jour de prévision J à 14h Paris, horizon h),
        avec la cible = consommation réelle à l'heure h du jour J+1.

        scenario = "perfect" : la météo utilisée en feature est celle
        RÉELLEMENT OBSERVÉE à l'heure visée (fuite assumée) -- sert de borne haute optimiste.

        scenario = "realistic" : on n'a pas de vraies prévisions météo
        historiques (seulement des observations, cf. synop_national_horaire.csv),
        donc on utilise comme proxy de "prévision" les valeurs météos connues À 14H LE JOUR J.
        Aucune fuite de données : à aucun moment on ne regarde une
        valeur postérieure à la coupure -- elle sert de borne réaliste basse,
        attendue moins bonne que le scénario "perfect".
        """

        if self.scenario not in ("perfect", "realistic"):
            raise ValueError(f"scenario inconnu : {self.scenario!r} (attendu 'perfect' ou 'realistic')")

        df = self._construct_useful_features(source_df)

        df_index = df.set_index(
            [df["timestamp_paris"].dt.normalize(), df["timestamp_paris"].dt.hour]
        )
        df_index.index.names = ["date_paris", "heure_paris"]

        jours = sorted(df["timestamp_paris"].dt.normalize().unique())

        final_jours = []
        for jour_j in jours:
            if (jour_j, self.pred_hour_local - 1) not in df_index.index:
                continue  # heure de coupure absente ce jour-là (bord du dataset)

            jour_j_heure_coupure = df_index.loc[(jour_j, self.pred_hour_local - 1)]
            if isinstance(jour_j_heure_coupure, pd.DataFrame):
                jour_j_heure_coupure = jour_j_heure_coupure.iloc[0]

            jour_j_plus_1 = jour_j + pd.DateOffset(days=1)

            # Toutes les heures qui existent réellement pour le jour J+1
            # (23, 24 ou 25 selon changement d'heure) : on énumère les lignes
            # du DataFrame source, pas un range(24) en dur.
            masque_j_plus_1 = df["timestamp_paris"].dt.normalize() == jour_j_plus_1
            jour_heures_j_plus_1 = df.loc[masque_j_plus_1].sort_values("timestamp_paris")

            for _, jour_heure_cible in jour_heures_j_plus_1.iterrows():
                cible_heure = jour_heure_cible["timestamp_paris"].hour

                ligne = {
                    "prevision_date": jour_j,
                    "cible_heure": cible_heure,
                    "cible_timestamp_paris": jour_heure_cible["timestamp_paris"],
                    "cible_timestamp_utc": jour_heure_cible["timestamp_utc"],
                    "cible_consommation_mw": jour_heure_cible["consommation_mw"],

                    "cible_heure_sin": np.sin(2 * np.pi * cible_heure / 24),
                    "cible_heure_cos": np.cos(2 * np.pi * cible_heure / 24),
                    "cible_jour_semaine_sin": np.sin(2 * np.pi * jour_heure_cible["jour_semaine"] / 7),
                    "cible_jour_semaine_cos": np.cos(2 * np.pi * jour_heure_cible["jour_semaine"] / 7),
                    "cible_jour_annee_sin": np.sin(2 * np.pi * jour_heure_cible["jour_annee"] / 365.25),
                    "cible_jour_annee_cos": np.cos(2 * np.pi * jour_heure_cible["jour_annee"] / 365.25),
                }

                # --- calendaire de l'heure visée (toujours connu à l'avance) ---
                for col in COLONNES_CALENDRIER:
                    ligne[col] = jour_heure_cible[col]

                # confinement_numero : NaN veut dire "pas en confinement", pas
                # "valeur manquante" -- on l'explicite en 0 pour éviter toute ambiguïté
                if pd.isna(ligne.get("confinement_numero")):
                    ligne["confinement_numero"] = 0

                for col in self.features_cols:
                    if col not in ligne and col in jour_heure_cible:
                        ligne[col] = jour_heure_cible[col]

                final_jours.append(ligne)

        table = pd.DataFrame(final_jours)
        table = table.dropna(subset=self.features_cols)

        self.verify_features(table)

        return table


    def verify_features(self, table):
        """Quelques contrôles de bon sens lancés après la construction de la table."""
        n_lignes = len(table)
        n_jours = table["prevision_date"].nunique()
        cible_manquante = int(table["cible_consommation_mw"].isna().sum())

        print("\nVérification de la table de features")
        print(f"Lignes totales : {n_lignes}")
        print(f"Jours de prévision uniques : {n_jours}")
        print(f"Cibles manquantes : {cible_manquante}")

        par_jour = table.groupby("prevision_date").size()
        jours_anormaux = par_jour[~par_jour.isin([23, 24, 25])]

        if not jours_anormaux.empty:
            print("\nATTENTION : Jours avec un nombre d'horizons anormal (!= 23, 24, 25) :")
            print(jours_anormaux)
        else:
            print("Aucune anomalie d'horizons détectée (tous les jours ont 23, 24 ou 25 heures).")


    def split_data(self, val_start, test_start, load_existing_splits=False):
        if load_existing_splits and \
            (PROCESSED_DIR / "train.csv").exists() and \
            (PROCESSED_DIR / "val.csv").exists() and \
            (PROCESSED_DIR / "test.csv").exists():
            return self.load_splits()

        feature_table = self.construct_features(self.df)

        train = feature_table[feature_table[self.date_col] < val_start].copy()
        val = feature_table[(feature_table[self.date_col] >= val_start) & (feature_table[self.date_col] < test_start)].copy()
        test = feature_table[feature_table[self.date_col] >= test_start].copy()

        print(f"Découpage Temporel")
        print(
            f"Train: {len(train)} lignes ({len(train) / len(feature_table) * 100:.1f}%)\n"
            f"Val: {len(val)} lignes ({len(val) / len(feature_table) * 100:.1f}%)\n"
            f"Test: {len(test)} lignes ({len(test) / len(feature_table) * 100:.1f}%)\n"
        )

        train.to_csv(PROCESSED_DIR / "train.csv", index=False)
        val.to_csv(PROCESSED_DIR / "val.csv", index=False)
        test.to_csv(PROCESSED_DIR / "test.csv", index=False)

        self.df_train = train
        self.df_val = val
        self.df_test = test

        return train, val, test


    def load_splits(self):
        train = pd.read_csv(PROCESSED_DIR / "train.csv")
        val = pd.read_csv(PROCESSED_DIR / "val.csv")
        test = pd.read_csv(PROCESSED_DIR / "test.csv")

        for df in (train, val, test):
            df['cible_timestamp_paris'] = pd.to_datetime(df['cible_timestamp_paris'], utc=True).dt.tz_convert('Europe/Paris')
            df['cible_timestamp_utc'] = pd.to_datetime(df['cible_timestamp_utc'], utc=True)

        self.df_train = train
        self.df_val = val
        self.df_test = test

        return train, val, test


    def evaluate_metrics(self, y_true, y_pred, hours=None):
        """MAE/RMSE/MAPE globaux. Si hours est fourni, renvoie aussi le détail par heure."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        resultats = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        }

        if hours is not None and len(hours) == len(y_true):
            df_eval = pd.DataFrame({"cible_heure": hours, "y": y_true, "pred": y_pred})
            per_hour = df_eval.groupby("cible_heure").apply( # type: ignore
                lambda g: pd.Series({
                    "MAE": mean_absolute_error(g["y"], g["pred"]),
                    "RMSE": root_mean_squared_error(g["y"], g["pred"]),
                }),
                include_groups=False, # type: ignore
            )
            resultats["per_hour"] = per_hour

        return resultats


    def _rolling_validation(self, model, df_train, df_val, freq="6ME", eval_train=True) -> tuple:
        """
        Validation walk-forward unifiée.
        - df_train : Historique initial
        - df_val : Données sur lesquelles on va évaluer de manière glissante
        - freq : Fréquence de réentraînement (ex: "1D" pour quotidien, "7D" pour hebdomadaire)
        """
        model_name = model.__class__.__name__
        df_combined = pd.concat([df_train, df_val]).sort_values(self.date_col)

        start = df_val[self.date_col].dt.normalize().min()
        end = df_val[self.date_col].dt.normalize().max() + pd.DateOffset(days=1)

        # Génération des dates de coupure (cutoffs)
        cutoffs = list(pd.date_range(start=start, end=end, freq=freq))
        if len(cutoffs) == 0 or cutoffs[0] > start:
            cutoffs = [start] + cutoffs
        if cutoffs[-1] < end:
            cutoffs.append(end)
        cutoffs = pd.DatetimeIndex(cutoffs)

        print(f"Évaluation de {model_name} | {len(df_val)} lignes | réentraînement/{freq} | {len(cutoffs)-1} itérations")

        all_train_preds = []
        all_val_preds = []
        all_train_y = []
        all_val_y = []
        all_train_horizons = []
        all_val_horizons = []

        for i in range(len(cutoffs) - 1):
            date = cutoffs[i]
            next_date = cutoffs[i + 1]

            train = df_combined[df_combined[self.date_col] < date].copy()
            val = df_combined[(df_combined[self.date_col] >= date) & (df_combined[self.date_col] < next_date)].copy()

            if not val.empty:
                model.fit(train)

                if eval_train:
                    train_preds = model.predict(train)
                    all_train_preds.extend(train_preds)
                    all_train_y.extend(train[self.target_col].values)
                    all_train_horizons.extend(train['cible_heure'].values)

                val_preds = model.predict(val)
                all_val_preds.extend(val_preds)
                all_val_y.extend(val[self.target_col].values)
                all_val_horizons.extend(val['cible_heure'].values)

                sys.stdout.write(f"\rProgression: {date.strftime('%Y-%m-%d %H:%M')}...")
                sys.stdout.flush()

        if eval_train:
            result_train_metrics = self.evaluate_metrics(all_train_y, all_train_preds, hours=all_train_horizons)
            print(
                f"\nTrain : MAE = {result_train_metrics['MAE']:.2f} MW |"
                f"RMSE = {result_train_metrics['RMSE']:.2f} MW |"
                f"MAPE = {result_train_metrics['MAPE']:.2f}%\n"
            )

        result_val_metrics = self.evaluate_metrics(all_val_y, all_val_preds, hours=all_val_horizons)
        print(
            f"\nValidation : MAE = {result_val_metrics['MAE']:.2f} MW |"
            f"RMSE = {result_val_metrics['RMSE']:.2f} MW |"
            f"MAPE = {result_val_metrics['MAPE']:.2f}%\n"
        )

        return result_train_metrics, all_train_preds, result_val_metrics, all_val_preds

    def rolling_test(self, model, freq="6ME"):
        """
        Effectue une test glissante d'un modèle.
        Évalue un modèles sur les ensembles Train|Validation et Test
        """
        model_name = model.__class__.__name__

        print(f"Test évaluation de {model_name}")

        df_trainval = pd.concat([self.df_train, self.df_val]).sort_values(self.date_col)
        _, _, result_val_metrics, result_val_preds = \
            self._rolling_validation(model, df_trainval, self.df_test, freq=freq, eval_train=False)

        if not self.models.get(model_name):
            self.models[model_name] = { "model": model }
        self.models[model_name]["test_preds"] = result_val_preds
        self.models[model_name]["test_metrics"] = result_val_metrics

        return self.models.get(model_name)


    def grid_search_rolling_validation(self, model_class, param_grid, freq="6ME", metric='MAE'):
        """
        Recherche par grille des meilleurs hyperparamètres avec validation glissante.
        """
        best_score = float('inf')
        best_result = dict()
        model_name = model_class.__name__
        results = []

        grid = list(ParameterGrid(param_grid))
        print(f"Grid Search de {model_name} :\n")

        for i, params in enumerate(grid):
            print(f"Combination {i+1}/{len(grid)} : {params}")

            model = model_class(**params)

            result_train_metrics, result_train_preds, result_val_metrics, result_val_preds = \
                self._rolling_validation(model, self.df_train, self.df_val, freq=freq)
            score = result_val_metrics[metric]

            results.append({
                'params': params,
                'score': score,
                'train_metrics': result_train_metrics,
                'val_metrics': result_val_metrics,
                'train_preds': result_train_preds,
                'val_preds': result_val_preds
            })

            if score < best_score:
                best_score = score
                best_result = results[-1]

        print(f"Best {metric} sur Validation : {best_score:.2f} MW")
        print(f"Best Params : {best_result['params']}\n")

        best_model = model_class(**best_result['params'])
        best_result["model"] = best_model
        del best_result['params']
        del best_result['score']

        self.models[model_name] = best_result

        return results


    def plot_feature_importances(self, top_n=15):
        feature_importances = {}
        for model_name, model_results in self.models.items():
            model = model_results.get("model")
            if not model: continue
            
            df_imp = model.get_feature_importances()
            feature_importances[model_name] = df_imp

            if df_imp is None or df_imp.empty:
                continue

            plt.figure(figsize=(16, 6))
            df_plot = df_imp.head(top_n).copy()

            plt.barh(df_plot['Feature'][::-1], df_plot['Importance'][::-1], color='skyblue')
            plt.xlabel("Importance")
            plt.title(f"Importance des Features - {model_name}")
            plt.tight_layout()
            plt.show()


    def plot_train_val_test_metrics(self, metric='MAE'):
        """
        Compare visuellement la métrique Train vs Validation pour diagnostiquer le sur/sous-apprentissage.
        """
        model_names = []
        train_scores = []
        val_scores = []
        test_scores = []

        for name, model_result in self.models.items():
            train_score = model_result.get('train_metrics', {}).get(metric, 0)
            val_score = model_result.get('val_metrics', {}).get(metric, 0)
            test_score = model_result.get('test_metrics', {}).get(metric, 0)

            model_names.append(name)
            train_scores.append(train_score)
            val_scores.append(val_score)
            test_scores.append(test_score)

        x = np.arange(len(model_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(model_names)*2), 6))
        rects1 = ax.bar(x - width, train_scores, width, label='Train', color='green')
        rects2 = ax.bar(x, val_scores, width, label='Validation', color='blue')
        rects3 = ax.bar(x + width, test_scores, width, label='Test', color='red')

        ax.set_ylabel(metric)
        ax.set_title(f'Diagnostic ({metric})')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        fig.tight_layout()
        plt.show()


    def plot_predictions(self, start_date=None, end_date=None, use_test=True):
        if use_test:
            df_eval = self.df_test
            pred_key = "test_preds"
        else:
            df_eval = self.df_val
            pred_key = "val_preds"

        if df_eval is None:
            print("Aucune donnée disponible pour le tracé.")
            return

        mask = pd.Series(True, index=df_eval.index)
        if start_date:
            mask = mask & (df_eval[self.date_col] >= start_date)
        if end_date:
            mask = mask & (df_eval[self.date_col] <= end_date)

        df_plot = df_eval[mask]

        models = [name for name in self.models.keys() if pred_key in self.models[name]]
        n_models = len(models)

        if n_models == 0:
            print(f"Aucune prédiction '{pred_key}' disponible.")
            return

        fig, axes = plt.subplots(
            n_models,
            1,
            figsize=(16, 4 * n_models),
            sharex=True
        )

        if n_models == 1:
            axes = [axes]

        for ax, name in zip(axes, models):
            preds = self.models[name][pred_key]
            
            # preds is a list or array, we need to convert it to Series matching df_eval index
            if not isinstance(preds, pd.Series):
                preds = pd.Series(preds, index=df_eval.index)
                
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

        fig.suptitle("Comparaison des Modèles : Prédictions vs Réel", fontsize=12)
        plt.ylabel("Consommation (MW)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_error_per_horizon(self, metric='MAE', plot_test=True):
        """
        Trace l'erreur (ex: MAE, RMSE) en fonction de l'horizon (heure cible de la journÃ©e).
        Affiche les erreurs de Validation (pointillÃ©s) et Test (ligne pleine) si disponibles.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 7))

        model_plotted = False
        for model_name, model_dict in self.models.items():
            # 1. Validation
            val_metrics = model_dict.get('val_metrics', {})
            if 'per_hour' in val_metrics:
                df_val = val_metrics['per_hour']
                if metric in df_val.columns:
                    plt.plot(
                        df_val.index, df_val[metric],
                        marker='o', linestyle='--', alpha=0.6,
                        label=f"{model_name} (Val)"
                    )
                    model_plotted = True

            # 2. Test
            test_metrics = model_dict.get('test_metrics', {})
            if plot_test and 'per_hour' in test_metrics:
                df_test = test_metrics['per_hour']
                if metric in df_test.columns:
                    plt.plot(
                        df_test.index, df_test[metric],
                        marker='s', linestyle='-', linewidth=2,
                        label=f"{model_name} (Test)"
                    )
                    model_plotted = True

        if not model_plotted:
            print("Aucune donnÃ©e par horizon trouvÃ©e.")
            plt.close()
            return

        plt.title(f"Erreur ({metric}) par Horizon : Validation vs Test")
        plt.xlabel("Heure cible (0h Ã  23h)")
        plt.ylabel(f"Erreur {metric} (MW)")
        plt.xticks(range(24))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
