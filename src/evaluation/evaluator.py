import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time, sys, os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import ParameterGrid
from pathlib import Path

project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)


from src.models.utils import Regressor, get_lagged_local_value, get_latest_local_value


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Colonnes météo utilisées comme features (observées, utilisées différemment
# selon le scénario -- voir construire_table_horizons).
COLONNES_METEO = [
    "temperature_c", "temperature_point_rosee_c", "humidite_pct",
    "vent_direction_deg", "vent_vitesse_ms", "nebulosite", "precip_1h_mm",
    "temperature_c_pondere_pop", "temperature_point_rosee_c_pondere_pop",
    "humidite_pct_pondere_pop", "vent_direction_deg_pondere_pop",
    "vent_vitesse_ms_pondere_pop", "nebulosite_pondere_pop",
    "precip_1h_mm_pondere_pop",
]

# Colonnes calendaires de l'heure visée : toujours connues à l'avance,
# jamais de risque de fuite.
COLONNES_CALENDRIER = [
    "jour_semaine", "mois", "saison", "weekend", "ferie", "vacances",
    "confinement_numero",
]


class TimeSeriesEvaluator:
    """
    Classe universelle pour évaluer les modèles de prévision temporelle.
    Gère le découpage chronologique, l'entraînement, l'inférence et le calcul des métriques.
    """
    def __init__(self, df, target_col='consommation_mw', pred_hour_local=14, scenario="perfect", features_cols=[
        "consommation_mw_moins_1",
        "consommation_mw_moins_2",
        "consommation_mw_moins_7",
        "consommation_mw_moins_365",
        "consommation_mw_moins_366",
        "consommation_mw_meme_heure_derniere_connue",
        "consommation_mw_derniere_connue",
        "consommation_mw_dernieres_24h_moyenne_derniere_connue",
        "consommation_mw_dernieres_24h_min_derniere_connue",
        "consommation_mw_dernieres_24h_max_derniere_connue"
    ]):
        self.df = df.sort_values('timestamp_paris').copy()
        self.target_col = target_col
        self.date_col = 'timestamp_paris'
        self.pred_hour_local = pred_hour_local
        self.scenario = scenario
        self.features_cols = features_cols

        self.df['timestamp_utc'] = pd.to_datetime(self.df['timestamp_utc'], utc=True)
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], utc=True).dt.tz_convert('Europe/Paris')

        self.results = {}
        self.predictions = {}


    def _construct_lagged_features(self, df):
        # Calcul des rolling stats en premier pour qu'ils soient dispos pour la capture à 14:00
        roll = df["consommation_mw"].rolling(window=24, min_periods=12)
        df["consommation_mw_dernieres_24h_moyenne_derniere_connue"] = roll.mean()
        df["consommation_mw_dernieres_24h_min_derniere_connue"] = roll.min()
        df["consommation_mw_dernieres_24h_max_derniere_connue"] = roll.max()

        df['date'] = df['timestamp_paris'].dt.date

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
        cutoff['date'] = (pd.to_datetime(cutoff['date']) + pd.Timedelta(days=1)).dt.date

        for col in lastest_known_cols:
            if col in cutoff.columns:
                col_cutoff = cutoff.set_index('date')[col]
                df[f'{col}_derniere_connue'] = df['date'].map(col_cutoff)

        for col in lastest_known_same_hour_cols:
            df[f'{col}_meme_heure_derniere_connue'] = get_latest_local_value(df, df, col, self.pred_hour_local)

        return df


    def construct_features(self):
        """Construit la table d'entraînement/évaluation :
        une ligne par (jour de prévision J à 14h Paris, horizon h de 0 à 23),
        avec la cible = consommation réelle à l'heure h du jour J+1.

        scenario = "perfect" : la météo utilisée en feature est celle
        RÉELLEMENT OBSERVÉE à l'heure visée (fuite assumée, voir docstring du
        module) -- sert de borne haute optimiste.

        scenario = "realistic" : on n'a pas de vraies prévisions météo
        historiques (seulement des observations, cf. synop_national_horaire.csv),
        donc on utilise comme proxy de "prévision" la DERNIÈRE VALEUR MÉTÉO
        CONNUE À 14H LE JOUR J, persistée telle quelle pour les 24 heures de
        J+1 (même principe que consommation_mw_moins_1_meme_heure_derniere_connue, mais appliqué à la
        météo). Aucune fuite de données : à aucun moment on ne regarde une
        valeur postérieure à la coupure. C'est une approximation volontairement
        grossière (une vraie prévision météo anticiperait un changement de
        temps, une persistance non) -- elle sert de borne réaliste basse,
        attendue moins bonne que le scénario "perfect".
        """

        if self.scenario not in ("perfect", "realistic"):
            raise ValueError(f"scenario inconnu : {self.scenario!r} (attendu 'perfect' ou 'realistic')")

        self._construct_lagged_features(self.df)

        df_index = self.df.set_index(
            [self.df["timestamp_paris"].dt.normalize(), self.df["timestamp_paris"].dt.hour]
        )
        df_index.index.names = ["date_paris", "heure_paris"]

        jours_possibles = sorted(self.df["timestamp_paris"].dt.normalize().unique())

        lignes = []
        for jour_j in jours_possibles:
            if (jour_j, self.pred_hour_local) not in df_index.index:
                continue  # heure de coupure absente ce jour-là (bord du dataset)

            ligne_coupure = df_index.loc[(jour_j, self.pred_hour_local)]
            if isinstance(ligne_coupure, pd.DataFrame):
                ligne_coupure = ligne_coupure.iloc[0]

            jour_j_plus_1 = jour_j + pd.Timedelta(days=1)

            # Toutes les heures qui existent réellement pour le jour J+1
            # (23, 24 ou 25 selon changement d'heure) : on énumère les lignes
            # du DataFrame source, pas un range(24) en dur.
            masque_j_plus_1 = self.df["timestamp_paris"].dt.normalize() == jour_j_plus_1
            heures_j_plus_1 = self.df.loc[masque_j_plus_1].sort_values("timestamp_paris")

            for _, ligne_cible in heures_j_plus_1.iterrows():
                horizon_h = ligne_cible["timestamp_paris"].hour

                ligne = {
                    "date_prevision": jour_j,
                    "horizon_h": horizon_h,
                    "timestamp_cible_paris": ligne_cible["timestamp_paris"],
                    "timestamp_cible_utc": ligne_cible["timestamp_utc"],
                    "cible_consommation_mw": ligne_cible["consommation_mw"],

                    "heure_sin": np.sin(2 * np.pi * horizon_h / 24),
                    "heure_cos": np.cos(2 * np.pi * horizon_h / 24),
                    "jour_annee_sin": np.sin(2 * np.pi * ligne_cible["jour_annee"] / 365.25),
                    "jour_annee_cos": np.cos(2 * np.pi * ligne_cible["jour_annee"] / 365.25),
                }

                for col in self.features_cols:
                    ligne[col] = ligne_cible[col]

                # --- calendaire de l'heure visée (toujours connu à l'avance) ---
                for col in COLONNES_CALENDRIER:
                    ligne[col] = ligne_cible[col]

                # confinement_numero : NaN veut dire "pas en confinement", pas
                # "valeur manquante" -- on l'explicite en 0 pour éviter toute
                # ambiguïté avec les vrais NaN (manque d'historique) des lags.
                if pd.isna(ligne["confinement_numero"]):
                    ligne["confinement_numero"] = 0

                # --- météo : "parfaite" = observée à l'heure visée (fuite
                # assumée) ; "realiste" = dernière valeur connue à 14h le jour
                # J, persistée pour les 24h de J+1 (pas de fuite). Le préfixe
                # de colonne change selon le scénario, pour bien distinguer
                # quel fichier contient quoi et éviter toute confusion entre
                # les deux variantes.
                source_meteo = ligne_cible if self.scenario == "perfect" else ligne_coupure
                prefixe_meteo = "meteo_parfaite_" if self.scenario == "perfect" else "meteo_realiste_"
                for col in COLONNES_METEO:
                    ligne[f"{prefixe_meteo}{col}"] = source_meteo[col]

                # --- degrés-jours, motivés par la relation en U vue en EDA ---
                # Même logique : calculés sur la météo "parfaite" ou "realiste"
                # selon le scénario, jamais sur une valeur postérieure à la
                # coupure en scénario realiste.
                temp = source_meteo["temperature_c_pondere_pop"]
                ligne["degres_sous_15"] = max(0.0, 15 - temp) if pd.notna(temp) else np.nan
                ligne["degres_au_dessus_22"] = max(0.0, temp - 22) if pd.notna(temp) else np.nan

                lignes.append(ligne)

        table = pd.DataFrame(lignes)
        table = table.dropna(subset=self.features_cols)
        
        # Vérification de sécurité automatique
        self.verify_features(table)
        
        return table

    def verify_features(self, table):
        """Quelques contrôles de bon sens lancés après la construction de la table."""
        n_lignes = len(table)
        n_jours = table["date_prevision"].nunique()
        cible_manquante = int(table["cible_consommation_mw"].isna().sum())
        
        print("\n=== Vérification de la table de features ===")
        print(f"Lignes totales : {n_lignes}")
        print(f"Jours de prévision uniques : {n_jours}")
        print(f"Cibles manquantes : {cible_manquante}")
        
        par_jour = table.groupby("date_prevision").size()
        jours_anormaux = par_jour[~par_jour.isin([23, 24, 25])]
        
        if not jours_anormaux.empty:
            print("\nATTENTION : Jours avec un nombre d'horizons anormal (!= 23, 24, 25) :")
            print(jours_anormaux)
        else:
            print("Aucune anomalie d'horizons détectée (tous les jours ont 23, 24 ou 25 heures).")
            
        print("============================================\n")


    def split_data(self, val_start, test_start):
        self.df = self.construct_features()
        self.date_col = 'timestamp_cible_paris'
        self.target_col = 'cible_consommation_mw'

        train = self.df[self.df[self.date_col] < val_start].copy()
        val = self.df[(self.df[self.date_col] >= val_start) & (self.df[self.date_col] < test_start)].copy()
        test = self.df[self.df[self.date_col] >= test_start].copy()

        print(f"Découpage Temporel")
        print(
            f"Train: {len(train)} lignes ({len(train) / len(self.df) * 100:.1f}%)\n"
            f"Val: {len(val)} lignes ({len(val) / len(self.df) * 100:.1f}%)\n"
            f"Test: {len(test)} lignes ({len(test) / len(self.df) * 100:.1f}%)\n"
        )

        train.to_csv(PROCESSED_DIR / "train.csv", index=False)
        val.to_csv(PROCESSED_DIR / "val.csv", index=False)
        test.to_csv(PROCESSED_DIR / "test.csv", index=False)

        return train, val, test


    def load_splits(self):
        train = pd.read_csv(PROCESSED_DIR / "train.csv")
        val = pd.read_csv(PROCESSED_DIR / "val.csv")
        test = pd.read_csv(PROCESSED_DIR / "test.csv")

        self.date_col = 'timestamp_cible_paris'
        self.target_col = 'cible_consommation_mw'

        for df in (train, val, test):
            df['timestamp_cible_paris'] = pd.to_datetime(df['timestamp_cible_paris'], utc=True).dt.tz_convert('Europe/Paris')
            df['timestamp_cible_utc'] = pd.to_datetime(df['timestamp_cible_utc'], utc=True)

        return train, val, test


    def evaluate_metrics(self, y_true, y_pred, horizon_h=None):
        """MAE/RMSE/MAPE globaux. Si horizon_h est fourni, renvoie aussi le détail par horizon."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        resultats = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        }

        if horizon_h is not None:
            df_eval = pd.DataFrame({"horizon_h": horizon_h, "y": y_true, "pred": y_pred})
            par_horizon = df_eval.groupby("horizon_h").apply(
                lambda g: pd.Series({
                    "MAE": mean_absolute_error(g["y"], g["pred"]),
                    "RMSE": root_mean_squared_error(g["y"], g["pred"]),
                }),
                include_groups=False,
            )
            resultats["par_horizon"] = par_horizon

        return resultats

    def plot_feature_importances(self, model, feature_cols, top_n=20):
        """Affiche les top_n features les plus importantes d'un modèle (ex: XGBoost)."""
        if hasattr(model, 'modele') and hasattr(model.modele, 'feature_importances_'):
            importances = model.modele.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
        else:
            print(f"Le modèle {model.__class__.__name__} n'a pas d'attribut feature_importances_.")
            return

        imp_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        imp_series.sort_values(ascending=True).plot(kind='barh')
        plt.title(f"Importance des Features (Top {top_n}) - {model.__class__.__name__}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
        
        return imp_series

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
        all_horizons = []

        for i in range(len(cutoffs) - 1):
            current_date = cutoffs[i]
            next_date = cutoffs[i + 1]

            current_train = df_combined[df_combined[self.date_col] < current_date].copy()
            current_val = df_combined[(df_combined[self.date_col] >= current_date) & (df_combined[self.date_col] < next_date)].copy()

            if not current_val.empty:
                model.fit(current_train)
                preds = model.predict(current_val)

                all_preds.extend(preds)
                all_y.extend(current_val[self.target_col].values)
                if 'horizon_h' in current_val.columns:
                    all_horizons.extend(current_val['horizon_h'].values)

                sys.stdout.write(f"\rProgression: {current_date.strftime('%Y-%m-%d %H:%M')}...")
                sys.stdout.flush()

        horizon_h_arr = all_horizons if all_horizons else None
        res_metrics = self.evaluate_metrics(all_y, all_preds, horizon_h=horizon_h_arr)

        print(f"\n\nScore : MAE = {res_metrics['MAE']:.2f} MW | RMSE = {res_metrics['RMSE']:.2f} MW | MAPE = {res_metrics['MAPE']:.2f}%\n")

        self.results[model_name] = res_metrics
        self.predictions[model_name] = pd.Series(all_preds, index=df_val.index)

        return res_metrics


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