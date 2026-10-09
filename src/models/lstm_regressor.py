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

vecteur statique riche (calendrier J+1, météo dérivée, consommations
    J-6 et "veille" du jour cible) passé par un MLP ;
option `residu` : le réseau prédit un écart par rapport à la
    consommation J-6 (même jour de semaine) ;
ensemble de plusieurs graines (moyenne des prédictions) : réduit la
    variance, gain quasi systématique avec peu de données.
Perte L1 (= MAE), early stopping sur la validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import copy
import torch
import torch.nn as nn

from src.models.utils import Regressor

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
    "cible_consommation_mw",
    "temperature_c_pondere_pop_derniere_connue",
    # "humidite_pct_pondere_pop",
    # "vent_vitesse_ms_pondere_pop",
    # "nebulosite_pondere_pop",
    # "precip_1h_mm_pondere_pop",
]
NOMS_SEQ = COLONNES_SEQ_BRUTES + ["cible_heure_sin", "cible_heure_cos", "weekend", "ferie"]

NOMS_METEO_DERIVEE = [
    "temperature_c_pondere_pop_moyenne_24h_derniere_connue",
    "temperature_c_pondere_pop_min_24h_derniere_connue",
    "temperature_c_pondere_pop_max_24h_derniere_connue",
    "temperature_c_pondere_pop_moyenne_48h_derniere_connue",
    "temperature_c_pondere_pop_moyenne_72h_derniere_connue",
    "temperature_c_pondere_pop_tendance_6h_derniere_connue",
    "temperature_c_pondere_pop_tendance_24h_derniere_connue",
    "temperature_c_pondere_pop_ecart_normale_derniere_connue",
    "temperature_c_pondere_pop_sous_15_cumul_3j_derniere_connue",
]
N_CAL, N_MET, N_J7, N_J1 = 10, len(NOMS_METEO_DERIVEE), 24, 24
SL_CAL = slice(0, N_CAL)
SL_MET = slice(N_CAL, N_CAL + N_MET)
SL_J7 = slice(N_CAL + N_MET, N_CAL + N_MET + N_J7)
SL_J1 = slice(N_CAL + N_MET + N_J7, N_CAL + N_MET + N_J7 + N_J1)
N_STAT = N_CAL + N_MET + N_J7 + N_J1


def _flag(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).clip(0, 1)


def preparer_grille(df):
    g = df.copy()
    g["weekend"] = _flag(g["weekend"])
    g["ferie"] = _flag(g["ferie"])
    for c in COLONNES_SEQ_BRUTES[1:]:
        g[c] = g[c].ffill().bfill()
    return g


def calculer_scalers(g):
    train = g[g["annee"] <= ANNEE_MAX_TRAIN]
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
    for j, c in enumerate(["cible_heure_sin", "cible_heure_cos", "weekend", "ferie"]):
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
    cm, cs = sc["moy"]["cible_consommation_mw"], sc["std"]["cible_consommation_mw"]

    date = g["cible_timestamp_paris"].dt.normalize()
    heure = g["cible_timestamp_paris"].dt.hour

    jc = g.assign(_d=date).drop_duplicates("_d").set_index("_d")
    cal = {d: r for d, r in pd.DataFrame({
        "ferie": _flag(jc["ferie"]), "vacances": _flag(jc["vacances"]),
        "confinement": jc["confinement_numero"].notna().astype(float),
    }).iterrows()}

    piv = (g.assign(_d=date, _h=heure).drop_duplicates(["_d", "_h"])
             .pivot(index="_d", columns="_h", values="cible_consommation_mw")
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

        j7 = ligne(J1 - 7 * un)
        veille = np.where(np.arange(24) <= HEURE_COUPURE, ligne(J), ligne(J - un))

        # Les features tabulaires de NOMS_METEO_DERIVEE correspondent a la meteo du jour cible
        # Il faut donc les recuperer sur J1, et non pas sur p (qui est J 14:00) !
        p_j1_idx = np.where((date == J1).values)[0]
        if len(p_j1_idx) == 0:
            continue
        p_j1 = p_j1_idx[0]

        stat = np.concatenate([_calendrier(J1, cal), met[p_j1], j7, veille])

        Xs.append(M[p - LONGUEUR_FENETRE + 1: p + 1])
        Xt.append(stat); Y.append(y); D.append(J)

    return (np.asarray(Xs, dtype=np.float32), np.asarray(Xt, dtype=np.float32),
            np.asarray(Y, dtype=np.float32), pd.DatetimeIndex(D), sc)


def decouper(X_seq, X_stat, y, dates):
    an = np.asarray(dates.year)
    m = {"train": an <= ANNEE_MAX_TRAIN, "val": an == 2024, "test": an >= 2025}
    return {k: (X_seq[v], X_stat[v], y[v], dates[v]) for k, v in m.items()}


def desnormaliser(y_norm, sc):
    return y_norm * sc["std"]["cible_consommation_mw"] + sc["moy"]["cible_consommation_mw"]




def fixer_graine(graine=42):
    np.random.seed(graine)
    torch.manual_seed(graine)


class ModeleLSTM(nn.Module):
    def __init__(self, n_seq, n_stat, hidden=64, couches=1, dropout=0.2, residu=False):
        super().__init__()
        self.residu = residu
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=couches, batch_first=True,
                            dropout=dropout if couches > 1 else 0.0)
        self.stat = nn.Sequential(nn.Linear(n_stat, 64), nn.ReLU(), nn.Dropout(dropout))
        self.tete = nn.Sequential(
            nn.Linear(hidden + 64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 24),
        )

    def forward(self, x_seq, x_stat):
        _, (h, _) = self.lstm(x_seq)
        sortie = self.tete(torch.cat([h[-1], self.stat(x_stat)], dim=1))
        if self.residu:
            sortie = sortie + x_stat[:, SL_J7]
        return sortie


def _t(jeu):
    return (torch.from_numpy(jeu[0]), torch.from_numpy(jeu[1]), torch.from_numpy(jeu[2]))


def entrainer_lstm(jeu_train, jeu_val, hidden=64, couches=1, dropout=0.2, residu=False,
                   lr=1e-3, weight_decay=1e-4, batch=64, max_epochs=80,
                   patience=10, graine=42, verbose=True):
    fixer_graine(graine)
    Xs, Xt, Y = _t(jeu_train)
    Xs_v, Xt_v, Y_v = _t(jeu_val)

    modele = ModeleLSTM(Xs.shape[2], Xt.shape[1], hidden, couches, dropout, residu)
    opt = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    perte = nn.L1Loss()

    meilleur, etat, sans = np.inf, None, 0
    hist = {"train": [], "val": []}
    for ep in range(1, max_epochs + 1):
        modele.train()
        perm = torch.randperm(len(Xs))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            l = perte(modele(Xs[idx], Xt[idx]), Y[idx])
            l.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            opt.step()
            tot += l.item() * len(idx)
        modele.eval()
        with torch.no_grad():
            lv = perte(modele(Xs_v, Xt_v), Y_v).item()
        hist["train"].append(tot / len(Xs)); hist["val"].append(lv)
        sched.step(lv)
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"epoch {ep:3d} | L1 train {hist['train'][-1]:.4f} | L1 val {lv:.4f}")
        if lv < meilleur - 1e-5:
            meilleur, etat, sans = lv, copy.deepcopy(modele.state_dict()), 0
        else:
            sans += 1
            if sans >= patience:
                if verbose:
                    print(f"Early stopping à l'epoch {ep} (meilleure val L1 normalisée : {meilleur:.4f})")
                break
    modele.load_state_dict(etat)
    modele.eval()
    return modele, hist


def entrainer_ensemble(jeu_train, jeu_val, n_modeles=5, graine0=0, verbose=False, **kw):
    """Entraîne n_modeles réseaux avec des graines différentes."""
    modeles = []
    for k in range(n_modeles):
        m, _ = entrainer_lstm(jeu_train, jeu_val, graine=graine0 + k, verbose=verbose, **kw)
        modeles.append(m)
        print(f"modèle {k + 1}/{n_modeles} entraîné")
    return modeles


def predire(modeles, jeu, scalers):
    """Prédictions en MW (N, 24). `modeles` : un modèle ou une liste."""
    if not isinstance(modeles, (list, tuple)):
        modeles = [modeles]
    Xs, Xt = torch.from_numpy(jeu[0]), torch.from_numpy(jeu[1])
    with torch.no_grad():
        p = np.mean([m.eval()(Xs, Xt).numpy() for m in modeles], axis=0)
    return desnormaliser(p, scalers)


def evaluer(modeles, jeu, scalers):
    pred = predire(modeles, jeu, scalers)
    reel = desnormaliser(jeu[2], scalers)
    e = pred - reel
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mape": float(np.mean(np.abs(e / reel)) * 100)}, pred, reel


class LSTMRegressor(Regressor):
    def __init__(self, raw_data_path="data/processed/dataset_final.csv", n_modeles=5, max_epochs=80, **kwargs):
        self.raw_data_path = raw_data_path
        self.n_modeles = n_modeles
        self.max_epochs = max_epochs
        self.kwargs = kwargs
        self.models = None
        self.scalers = None
        self.df_raw = None

    def fit(self, df_train: pd.DataFrame):
        # On sauvegarde le dataframe d'entraînement comme historique brut !
        self.df_raw = df_train.copy()

        jours_train = pd.to_datetime(df_train['prevision_date'], utc=True).dt.tz_convert('Europe/Paris').dt.normalize().unique()

        split_idx = int(len(jours_train) * 0.9)
        jours_t = jours_train[:split_idx]
        jours_v = jours_train[split_idx:]

        jeu_train = construire_jeu(self.df_raw, jours=jours_t, avec_cible=True)
        self.scalers = jeu_train[4]

        jeu_val = construire_jeu(self.df_raw, scalers=self.scalers, jours=jours_v, avec_cible=True)

        self.models = entrainer_ensemble(jeu_train, jeu_val, n_modeles=self.n_modeles, max_epochs=self.max_epochs, **self.kwargs)

    def predict(self, df_test: pd.DataFrame):
        jours_test = pd.to_datetime(df_test['prevision_date'], utc=True).dt.tz_convert('Europe/Paris').dt.normalize().unique()

        # Concaténer history + gap (Day J) + test (Day J+1) pour que les features séquentielles se calculent bien
        gap = getattr(self, 'gap_data', pd.DataFrame())
        df_combined = pd.concat([self.df_raw, gap, df_test])

        jeu_test = construire_jeu(df_combined, scalers=self.scalers, jours=jours_test, avec_cible=False)

        preds_24h = predire(self.models, jeu_test, self.scalers)

        pred_dict = {}
        dates_J = jeu_test[3]
        for i, date_j in enumerate(dates_J):
            pred_dict[date_j] = preds_24h[i]

        def get_pred(row):
            date_j = pd.to_datetime(row['prevision_date'], utc=True).tz_convert('Europe/Paris').normalize()
            heure = int(row['cible_heure'])
            if date_j in pred_dict:
                return pred_dict[date_j][heure]
            return np.nan

        return df_test.apply(get_pred, axis=1).values