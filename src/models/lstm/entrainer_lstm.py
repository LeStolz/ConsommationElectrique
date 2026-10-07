"""
LSTM  : prévoit les 24 heures de J+1 à partir de 7 jours
d'historique connus à 14h le jour J.

Améliorations vs version simple :
  - vecteur statique riche (calendrier J+1, météo dérivée, consommations
    J-6 et "veille" du jour cible) passé par un MLP ;
  - option `residu` : le réseau prédit un écart par rapport à la
    consommation J-6 (même jour de semaine) ;
  - ensemble de plusieurs graines (moyenne des prédictions) : réduit la
    variance, gain quasi systématique avec peu de données.
Perte L1 (= MAE), early stopping sur la validation.
"""

import copy
import numpy as np
import torch
import torch.nn as nn

try:
    from .features_lstm import desnormaliser, SL_J7
except ImportError:                      # import direct (dossier ajouté au path)
    from features_lstm import desnormaliser, SL_J7


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