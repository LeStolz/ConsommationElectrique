"""
Construction du tableau de features pour le modèle XGBoost -- scénarios
"météo parfaite" et "météo réaliste".

UN SEUL modèle XGBoost, entraîné sur une table "empilée" où chaque ligne correspond a
 (jour de prévision J, horizon
h de 0 à 23) -> cible = consommation réelle à l'heure h du jour J+1.

Scénario "météo parfaite" (volontairement optimiste, sert de référence
haute) : on utilise la météo RÉELLEMENT OBSERVÉE à l'heure visée (J+1, h)
comme si elle avait été parfaitement prévue à 14h le jour J. C'est une
fuite de données assumée et documentée -- ce scénario ne sera jamais
utilisable en production, seulement comme borne haute de performance pour
juger le scénario "réaliste".

Scénario "météo réaliste" (borne basse honnête) : on n'a pas de vraies
prévisions météo historiques, seulement des observations (synop_national_
horaire.csv). On utilise donc comme proxy de "prévision" la DERNIÈRE VALEUR
MÉTÉO CONNUE À 14H LE JOUR J, persistée telle quelle pour les 24 heures de
J+1 -- même principe que conso_J_moins_1_meme_heure, mais appliqué à la
météo. Aucune fuite de données : à aucun moment on ne regarde une valeur
postérieure à la coupure. C'est une approximation volontairement grossière
(une vraie prévision météo anticiperait un changement de temps, une
persistance non), mais c'est la seule option honnête avec les données
disponibles.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Fichier dans src/models/xgboost/ -> parents[3] = racine du projet
BASE_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HEURE_COUPURE = 14  # 14h heure de Paris : moment où on lance la prévision

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


# ============================================================
# CHARGEMENT
# ============================================================

def charger_dataset_final(chemin=None):
    """Charge dataset_final.csv, en reconstruisant proprement les deux
    horodatages (utc pour l'indexation, paris pour toute la logique
    métier) et en triant chronologiquement."""

    chemin = chemin or (PROCESSED_DIR / "dataset_final.csv")
    df = pd.read_csv(chemin)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_paris"] = df["timestamp_utc"].dt.tz_convert("Europe/Paris")

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    return df



# FEATURES DE LAG (consommation passée, connue à 14h le jour J)

def construire_lags_conso(df):
    """Ajoute au DataFrame (indexé/trié par timestamp_utc, une ligne par
    heure, SANS trou) les colonnes de lag de consommation, calculées sur
    la grille horaire elle-même (donc toujours alignées sur la même
    heure locale d'un jour à l'autre, sauf les 2 jours de changement
    d'heure -- limite mineure, documentée, acceptée pour l'instant).
    """

    df = df.copy()

    df["conso_J_moins_1_brut"] = df["consommation_mw"].shift(24)
    df["conso_J_moins_2_brut"] = df["consommation_mw"].shift(48)
    df["conso_J_moins_7"] = df["consommation_mw"].shift(7 * 24)
    df["conso_J_moins_365"] = df["consommation_mw"].shift(365 * 24)
    df["conso_J_moins_366"] = df["consommation_mw"].shift(366 * 24)

    # Moyenne/min/max glissants des 24 dernières heures CONNUES à cet
    # instant (fenêtre qui se termine à la ligne elle-même, jamais après).
    roll = df["consommation_mw"].rolling(window=24, min_periods=12)
    df["conso_dernieres_24h_moyenne"] = roll.mean()
    df["conso_dernieres_24h_min"] = roll.min()
    df["conso_dernieres_24h_max"] = roll.max()

    return df


def valeur_a_la_coupure(df, colonne, jour_paris, heure_coupure=HEURE_COUPURE):
    """Renvoie la valeur de `colonne` à la ligne (jour_paris, heure_coupure)
    -- c'est-à-dire la toute dernière valeur connue à 14h ce jour-là.
    `jour_paris` est une date (pas un timestamp)."""

    masque = (df["timestamp_paris"].dt.normalize() == jour_paris) & \
             (df["timestamp_paris"].dt.hour == heure_coupure)
    valeurs = df.loc[masque, colonne]
    if len(valeurs) == 0:
        return np.nan
    return valeurs.iloc[0]


# CONSTRUCTION DE LA TABLE "EMPILÉE" (1 ligne = 1 jour de prévision x 1 horizon)

def construire_table_horizons(df, scenario="parfait"):
    """Construit la table d'entraînement/évaluation pour XGBoost :
    une ligne par (jour de prévision J à 14h Paris, horizon h de 0 à 23),
    avec la cible = consommation réelle à l'heure h du jour J+1.

    scenario = "parfait" : la météo utilisée en feature est celle
    RÉELLEMENT OBSERVÉE à l'heure visée (fuite assumée, voir docstring du
    module) -- sert de borne haute optimiste.

    scenario = "realiste" : on n'a pas de vraies prévisions météo
    historiques (seulement des observations, cf. synop_national_horaire.csv),
    donc on utilise comme proxy de "prévision" la DERNIÈRE VALEUR MÉTÉO
    CONNUE À 14H LE JOUR J, persistée telle quelle pour les 24 heures de
    J+1 (même principe que conso_J_moins_1_meme_heure, mais appliqué à la
    météo). Aucune fuite de données : à aucun moment on ne regarde une
    valeur postérieure à la coupure. C'est une approximation volontairement
    grossière (une vraie prévision météo anticiperait un changement de
    temps, une persistance non) -- elle sert de borne réaliste basse,
    attendue moins bonne que le scénario "parfait".
    """

    if scenario not in ("parfait", "realiste"):
        raise ValueError(f"scenario inconnu : {scenario!r} (attendu 'parfait' ou 'realiste')")

    df = construire_lags_conso(df)

    # Index pratique pour retrouver n'importe quelle heure par (date, heure)
    df_index = df.set_index(
        [df["timestamp_paris"].dt.normalize(), df["timestamp_paris"].dt.hour]
    )
    df_index.index.names = ["date_paris", "heure_paris"]

    # Jours J candidats comme "jour de prévision" : il faut qu'il existe
    # une ligne (J, 14h) dans les données.
    jours_possibles = sorted(df["timestamp_paris"].dt.normalize().unique())

    lignes = []

    for jour_j in jours_possibles:

        if (jour_j, HEURE_COUPURE) not in df_index.index:
            continue  # heure de coupure absente ce jour-là (bord du dataset)

        ligne_coupure = df_index.loc[(jour_j, HEURE_COUPURE)]
        if isinstance(ligne_coupure, pd.DataFrame):
            ligne_coupure = ligne_coupure.iloc[0]

        jour_j_plus_1 = jour_j + pd.Timedelta(days=1)

        # Toutes les heures qui existent réellement pour le jour J+1
        # (23, 24 ou 25 selon changement d'heure) : on énumère les lignes
        # du DataFrame source, pas un range(24) en dur.
        masque_j_plus_1 = df["timestamp_paris"].dt.normalize() == jour_j_plus_1
        heures_j_plus_1 = df.loc[masque_j_plus_1].sort_values("timestamp_paris")

        for _, ligne_cible in heures_j_plus_1.iterrows():

            horizon_h = ligne_cible["timestamp_paris"].hour

            ligne = {
                "date_prevision": jour_j,
                "horizon_h": horizon_h,
                "timestamp_cible_paris": ligne_cible["timestamp_paris"],
                "timestamp_cible_utc": ligne_cible["timestamp_utc"],

                # --- cible ---
                "cible_consommation_mw": ligne_cible["consommation_mw"],

                # --- lags de conso, connus à 14h le jour J ---
                "conso_J_moins_7": ligne_cible["conso_J_moins_7"],
                "conso_J_moins_365": ligne_cible["conso_J_moins_365"],
                "conso_J_moins_366": ligne_cible["conso_J_moins_366"],
                "conso_dernieres_24h_moyenne": ligne_coupure["conso_dernieres_24h_moyenne"],
                "conso_dernieres_24h_min": ligne_coupure["conso_dernieres_24h_min"],
                "conso_dernieres_24h_max": ligne_coupure["conso_dernieres_24h_max"],

                # --- encodages cycliques ---
                "heure_sin": np.sin(2 * np.pi * horizon_h / 24),
                "heure_cos": np.cos(2 * np.pi * horizon_h / 24),
                "jour_annee_sin": np.sin(2 * np.pi * ligne_cible["jour_annee"] / 365.25),
                "jour_annee_cos": np.cos(2 * np.pi * ligne_cible["jour_annee"] / 365.25),
            }

            # "Même heure, jour précédent" : si cette heure de J est déjà
            # passée à 14h (donc hour <= 14), elle est connue -> J-1.
            # Sinon (hour > 14, ex. 18h), on ne la connaît pas encore à
            # l'instant de la coupure -> repli sur J-2.
            if horizon_h <= HEURE_COUPURE:
                ligne["conso_J_moins_1_meme_heure"] = ligne_cible["conso_J_moins_1_brut"]
            else:
                ligne["conso_J_moins_1_meme_heure"] = ligne_cible["conso_J_moins_2_brut"]

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
            source_meteo = ligne_cible if scenario == "parfait" else ligne_coupure
            prefixe_meteo = "meteo_parfaite_" if scenario == "parfait" else "meteo_realiste_"
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
    return table


# SPLIT TRAIN / VALIDATION / TEST 

def split_train_val_test(table):
    """Découpe temporelle sur date_prevision (jour J, pas le jour cible) :
    Train 2016-2023, Validation 2024, Test 2025 -> mi-2026."""

    date_prevision = pd.to_datetime(table["date_prevision"])

    train = table[date_prevision.dt.year <= 2023].copy()
    val = table[date_prevision.dt.year == 2024].copy()
    test = table[date_prevision.dt.year >= 2025].copy()

    return train, val, test

def predire_jour_suivant(df, modele, colonnes_features, jour_j, scenario="realiste"):
    """Prédit les 24 (23/25) heures de jour_j+1, sans exiger que ces heures
    existent déjà dans df -- seule l'historique jusqu'à 14h le jour_j est
    nécessaire."""

    if scenario != "realiste":
        raise ValueError("predire_jour_suivant ne gère que le scénario 'realiste' "
                         "(la météo de J+1 n'est pas connue à 14h).")

    jour_j = pd.Timestamp(jour_j)
    if jour_j.tzinfo is None:
        jour_j = jour_j.tz_localize("Europe/Paris")
    jour_j = jour_j.normalize()

    df = construire_lags_conso(df)

    df_index = df.set_index(
        [df["timestamp_paris"].dt.normalize(), df["timestamp_paris"].dt.hour]
    )
    df_index.index.names = ["date_paris", "heure_paris"]

    if (jour_j, HEURE_COUPURE) not in df_index.index:
        raise ValueError(f"Pas de ligne (jour_j, 14h) pour {jour_j} dans df.")

    ligne_coupure = df_index.loc[(jour_j, HEURE_COUPURE)]
    if isinstance(ligne_coupure, pd.DataFrame):
        ligne_coupure = ligne_coupure.iloc[0]

    jour_j_plus_1 = jour_j + pd.Timedelta(days=1)

    # On génère nous-mêmes les timestamps cibles -- pas besoin qu'ils
    # existent dans df. tz-aware, gère nativement les jours à 23/25h.
    debut = jour_j_plus_1
    fin = jour_j_plus_1 + pd.Timedelta(days=1)
    timestamps_cibles = pd.date_range(debut, fin, freq="h", tz="Europe/Paris", inclusive="left")

    lignes = []
    for ts in timestamps_cibles:
        horizon_h = ts.hour

        # Lags conso : on les récupère dans l'historique déjà connu, via
        # décalage depuis jour_j_plus_1 (pas besoin que ts existe dans df).
        def lag(colonne, jours):
            cible_lag = (ts - pd.Timedelta(days=jours)).normalize()
            heure_lag = (ts - pd.Timedelta(days=jours)).hour
            if (cible_lag, heure_lag) not in df_index.index:
                return np.nan
            l = df_index.loc[(cible_lag, heure_lag)]
            if isinstance(l, pd.DataFrame):
                l = l.iloc[0]
            return l[colonne]

        ligne = {
            "horizon_h": horizon_h,
            "conso_J_moins_7": lag("consommation_mw", 7),
            "conso_J_moins_365": lag("consommation_mw", 365),
            "conso_J_moins_366": lag("consommation_mw", 366),
            "conso_dernieres_24h_moyenne": ligne_coupure["conso_dernieres_24h_moyenne"],
            "conso_dernieres_24h_min": ligne_coupure["conso_dernieres_24h_min"],
            "conso_dernieres_24h_max": ligne_coupure["conso_dernieres_24h_max"],
            "heure_sin": np.sin(2 * np.pi * horizon_h / 24),
            "heure_cos": np.cos(2 * np.pi * horizon_h / 24),
            "jour_annee_sin": np.sin(2 * np.pi * ts.dayofyear / 365.25),
            "jour_annee_cos": np.cos(2 * np.pi * ts.dayofyear / 365.25),
            "conso_J_moins_1_meme_heure": lag("consommation_mw", 1 if horizon_h <= HEURE_COUPURE else 2),
            "jour_semaine": ts.dayofweek,
            "mois": ts.month,
            "saison": "Hiver" if ts.month in (12, 1, 2) else "Printemps" if ts.month in (3, 4, 5) else "Été" if ts.month in (6, 7, 8) else "Automne",
            "weekend": int(ts.dayofweek >= 5),
            "ferie": 0,
            "vacances": 0,
            "confinement_numero": 0,
        }

        # Calendrier (férié/vacances) : repris du dataset si la ligne
        # cible existe déjà (backtest) ; sinon 0 par défaut -> pour un vrai
        # futur, brancher un calendrier de fériés/vacances.
        if (ts.normalize(), horizon_h) in df_index.index:
            l_cal = df_index.loc[(ts.normalize(), horizon_h)]
            if isinstance(l_cal, pd.DataFrame):
                l_cal = l_cal.iloc[0]
            for c in ("ferie", "vacances"):
                ligne[c] = int(l_cal[c]) if pd.notna(l_cal[c]) else 0

        for col in COLONNES_METEO:
            ligne[f"meteo_realiste_{col}"] = ligne_coupure[col]

        temp = ligne_coupure["temperature_c_pondere_pop"]
        ligne["degres_sous_15"] = max(0.0, 15 - temp) if pd.notna(temp) else np.nan
        ligne["degres_au_dessus_22"] = max(0.0, temp - 22) if pd.notna(temp) else np.nan

        ligne["timestamp_cible_paris"] = ts
        lignes.append(ligne)

    table = pd.DataFrame(lignes)
    table = pd.get_dummies(table, columns=["saison"])
    for col in colonnes_features:
        if col.startswith("saison_") and col not in table.columns:
            table[col] = False

    X = table[colonnes_features].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    predictions = modele.predict(X)

    return pd.DataFrame({
        "timestamp_cible_paris": table["timestamp_cible_paris"],
        "horizon_h": table["horizon_h"],
        "consommation_predite_mw": predictions,
    }).reset_index(drop=True)


# ============================================================
# VÉRIFICATIONS DE SÉCURITÉ (pas de fuite, pas de ligne incomplète)
# ============================================================

def verifier_table(table):
    """Quelques contrôles de bon sens à lancer après construction."""

    resultats = {}

    resultats["n_lignes"] = len(table)
    resultats["n_jours_prevision"] = table["date_prevision"].nunique()

    # Un jour normal donne 24 lignes, les 2 jours de changement d'heure
    # en donnent 23 ou 25 -- jamais autre chose.
    par_jour = table.groupby("date_prevision").size()
    resultats["repartition_nb_horizons_par_jour"] = par_jour.value_counts().to_dict()
    resultats["jours_hors_23_24_25"] = par_jour[~par_jour.isin([23, 24, 25])].to_dict()

    # La cible ne doit jamais être manquante
    resultats["cible_manquante"] = int(table["cible_consommation_mw"].isna().sum())

    return resultats