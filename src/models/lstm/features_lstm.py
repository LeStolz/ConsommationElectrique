"""
Construction des séquences pour le LSTM .

Une observation = un jour de prévision J, lancé à 14h (heure de Paris).

ENTRÉES (rien de postérieur à 14h le jour J) :
  - séquence : les 168 dernières heures (7 jours) -- consommation, météo
    pondérée, heure, week-end, férié ;
  - statiques, concaténées en un seul vecteur de 67 valeurs :
      [0:10]   calendrier du jour cible J+1 (connu à l'avance)
      [10:19]  météo dérivée à 14h (moyennes 24/48/72h, min/max, tendances,
               écart à la normale saisonnière, froid cumulé) -- mêmes idées
               que les features XGBoost v2
      [19:43]  consommation du même jour J-6 (= J+1 moins 7 jours), 24 heures
      [43:67]  consommation "veille" : heures <=14 -> jour J, sinon J-1.
SORTIE : les 24 consommations horaires de J+1 (prédiction directe).

Découpes temporelles : train <= 2023, val 2024, test >= 2025 (année de J).
Toutes les normalisations sont apprises sur le train uniquement.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _racine_projet():
    """Remonte les dossiers jusqu'à trouver data/processed (robuste au
    dossier où ce fichier est rangé)."""
    for p in Path(__file__).resolve().parents:
        if (p / "data" / "processed").exists():
            return p
    return Path(__file__).resolve().parents[3]


BASE_DIR = _racine_projet()
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HEURE_COUPURE = 14
LONGUEUR_FENETRE = 168
ANNEE_MAX_TRAIN = 2023

COLONNES_SEQ_BRUTES = [
    "consommation_mw",
    "temperature_c_pondere_pop", "humidite_pct_pondere_pop",
    "vent_vitesse_ms_pondere_pop", "nebulosite_pondere_pop",
    "precip_1h_mm_pondere_pop",
]
NOMS_SEQ = COLONNES_SEQ_BRUTES + ["heure_sin", "heure_cos", "weekend", "ferie"]

NOMS_METEO_DERIVEE = [
    "temp_moy_24h", "temp_min_24h", "temp_max_24h", "temp_moy_48h",
    "temp_moy_72h", "temp_tendance_6h", "temp_tendance_24h",
    "temp_ecart_normale", "degres_sous_15_cumul_3j",
]
N_CAL, N_MET, N_J7, N_J1 = 10, len(NOMS_METEO_DERIVEE), 24, 24
SL_CAL = slice(0, N_CAL)
SL_MET = slice(N_CAL, N_CAL + N_MET)
SL_J7 = slice(N_CAL + N_MET, N_CAL + N_MET + N_J7)
SL_J1 = slice(N_CAL + N_MET + N_J7, N_CAL + N_MET + N_J7 + N_J1)
N_STAT = N_CAL + N_MET + N_J7 + N_J1


def charger_dataset(chemin=None):
    """Charge dataset_final.csv avec timestamp_utc / timestamp_paris."""
    chemin = chemin or (PROCESSED_DIR / "dataset_final.csv")
    df = pd.read_csv(chemin)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_paris"] = df["timestamp_utc"].dt.tz_convert("Europe/Paris")
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def _flag(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).clip(0, 1)


def _normale_saisonniere(g):
    t = g["temperature_c_pondere_pop"]
    annee = g["timestamp_paris"].dt.year
    jda = g["timestamp_paris"].dt.dayofyear
    m = annee <= ANNEE_MAX_TRAIN
    moy = t[m].groupby(jda[m]).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = pd.concat([moy.iloc[-15:], moy, moy.iloc[:15]])
    lisse = ext.rolling(31, center=True, min_periods=1).mean().iloc[15:-15]
    lisse.index = moy.index
    return jda.map(lisse)


def preparer_grille(df):
    g = df.sort_values("timestamp_utc").reset_index(drop=True).copy()
    if not (g["timestamp_utc"].diff().dropna() == pd.Timedelta(hours=1)).all():
        print("ATTENTION : la grille horaire a des trous.")
    h = g["timestamp_paris"].dt.hour
    g["heure_sin"] = np.sin(2 * np.pi * h / 24)
    g["heure_cos"] = np.cos(2 * np.pi * h / 24)
    g["weekend"] = _flag(g["weekend"])
    g["ferie"] = _flag(g["ferie"])
    for c in COLONNES_SEQ_BRUTES[1:]:
        g[c] = g[c].ffill().bfill()

    # météo dérivée (passé uniquement : fenêtres qui finissent à la ligne)
    t = g["temperature_c_pondere_pop"]
    g["temp_moy_24h"] = t.rolling(24, min_periods=12).mean()
    g["temp_min_24h"] = t.rolling(24, min_periods=12).min()
    g["temp_max_24h"] = t.rolling(24, min_periods=12).max()
    g["temp_moy_48h"] = t.rolling(48, min_periods=24).mean()
    g["temp_moy_72h"] = t.rolling(72, min_periods=36).mean()
    g["temp_tendance_6h"] = t - t.shift(6)
    g["temp_tendance_24h"] = t - t.shift(24)
    g["temp_ecart_normale"] = g["temp_moy_24h"] - _normale_saisonniere(g)
    g["degres_sous_15_cumul_3j"] = (15 - t).clip(lower=0).rolling(72, min_periods=36).mean()
    return g


def calculer_scalers(g):
    train = g[g["timestamp_paris"].dt.year <= ANNEE_MAX_TRAIN]
    cols = COLONNES_SEQ_BRUTES
    return {
        "moy": train[cols].mean().to_dict(),
        "std": train[cols].std().replace(0, 1).to_dict(),
        "met_moy": train[NOMS_METEO_DERIVEE].mean().to_dict(),
        "met_std": train[NOMS_METEO_DERIVEE].std().replace(0, 1).to_dict(),
    }


def _matrice_sequence(g, sc):
    M = np.zeros((len(g), len(NOMS_SEQ)), dtype=np.float32)
    for i, c in enumerate(COLONNES_SEQ_BRUTES):
        M[:, i] = ((g[c] - sc["moy"][c]) / sc["std"][c]).fillna(0).values
    k = len(COLONNES_SEQ_BRUTES)
    for j, c in enumerate(["heure_sin", "heure_cos", "weekend", "ferie"]):
        M[:, k + j] = g[c].values
    return M


def _calendrier(jour, cal):
    js = jour.dayofweek
    r = cal.get(jour)
    ferie, vac, conf = (float(r["ferie"]), float(r["vacances"]), float(r["confinement"])) if r is not None else (0.0, 0.0, 0.0)
    return [
        np.sin(2 * np.pi * js / 7), np.cos(2 * np.pi * js / 7),
        np.sin(2 * np.pi * (jour.month - 1) / 12), np.cos(2 * np.pi * (jour.month - 1) / 12),
        np.sin(2 * np.pi * jour.dayofyear / 365.25), np.cos(2 * np.pi * jour.dayofyear / 365.25),
        float(js >= 5), ferie, vac, conf,
    ]


def construire_jeu(df, scalers=None, jours=None, avec_cible=True):
    """Retourne (X_seq, X_stat, y, dates_J, scalers).
    jours=None : tous les jours possibles. avec_cible=False : inférence
    (y = NaN). Les jours dont J+1 n'a pas 24 heures sont écartés quand
    avec_cible=True."""
    g = preparer_grille(df)
    sc = scalers or calculer_scalers(g)
    M = _matrice_sequence(g, sc)
    cm, cs = sc["moy"]["consommation_mw"], sc["std"]["consommation_mw"]

    date = g["timestamp_paris"].dt.normalize()
    heure = g["timestamp_paris"].dt.hour

    jc = g.assign(_d=date).drop_duplicates("_d").set_index("_d")
    cal = {d: r for d, r in pd.DataFrame({
        "ferie": _flag(jc["ferie"]), "vacances": _flag(jc["vacances"]),
        "confinement": jc["confinement_numero"].notna().astype(float),
    }).iterrows()}

    piv = (g.assign(_d=date, _h=heure).drop_duplicates(["_d", "_h"])
             .pivot(index="_d", columns="_h", values="consommation_mw")
             .reindex(columns=range(24)))
    piv_n = (piv - cm) / cs

    met = ((g[NOMS_METEO_DERIVEE] - pd.Series(sc["met_moy"])) / pd.Series(sc["met_std"])
           ).fillna(0).values.astype(np.float32)

    pos14 = np.where((heure == HEURE_COUPURE).values)[0]
    pos_par_jour = {date.iloc[p]: p for p in pos14}
    if jours is None:
        jours = list(pos_par_jour)
    un = pd.Timedelta(days=1)

    def ligne(jour, h_ok=None):
        if jour not in piv_n.index:
            return np.zeros(24)
        v = piv_n.loc[jour].values.astype(float)
        return np.nan_to_num(v, nan=0.0)

    Xs, Xt, Y, D = [], [], [], []
    for J in jours:
        p = pos_par_jour.get(J)
        if p is None or p < LONGUEUR_FENETRE - 1:
            continue
        J1 = J + un
        if avec_cible:
            if J1 not in piv.index or piv.loc[J1].isna().any():
                continue
            y = piv_n.loc[J1].values
        else:
            y = np.full(24, np.nan)

        j7 = ligne(J1 - 7 * un)                       # J-6, entièrement connu
        veille = np.where(np.arange(24) <= HEURE_COUPURE, ligne(J), ligne(J - un))
        stat = np.concatenate([_calendrier(J1, cal), met[p], j7, veille])

        Xs.append(M[p - LONGUEUR_FENETRE + 1: p + 1])
        Xt.append(stat); Y.append(y); D.append(J)

    return (np.asarray(Xs, dtype=np.float32), np.asarray(Xt, dtype=np.float32),
            np.asarray(Y, dtype=np.float32), pd.DatetimeIndex(D), sc)


def decouper(X_seq, X_stat, y, dates):
    an = np.asarray(dates.year)
    m = {"train": an <= ANNEE_MAX_TRAIN, "val": an == 2024, "test": an >= 2025}
    return {k: (X_seq[v], X_stat[v], y[v], dates[v]) for k, v in m.items()}


def desnormaliser(y_norm, sc):
    return y_norm * sc["std"]["consommation_mw"] + sc["moy"]["consommation_mw"]