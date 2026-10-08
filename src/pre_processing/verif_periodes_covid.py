"""
Vérification visuelle des 3 périodes de confinement national.

Objectif  : ne pas se contenter des dates officielles des
décrets, mais VISUALISER la conso autour de chaque confinement (avec des
semaines de référence avant/après) pour juger si la rupture dans les
données colle bien à ces dates, ou si elle commence/finit en décalage.

"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PERIODES_COVID = [
    ("Confinement 1", "2020-03-17", "2020-05-11"),
    ("Confinement 2", "2020-10-30", "2020-12-15"),
    ("Confinement 3", "2021-04-03", "2021-05-03"),
]

MARGE_JOURS = 21  

def charger():
    chemin = PROCESSED_DIR / "dataset_final.csv"
    df = pd.read_csv(chemin)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    df.index = df.index.tz_convert("Europe/Paris")

    conso_jour = df["consommation_mw"].resample("D").mean()
    return conso_jour


def zoom_periode(conso_jour, nom, debut, fin):
  
    debut = pd.Timestamp(debut, tz="Europe/Paris")
    fin = pd.Timestamp(fin, tz="Europe/Paris")
    fenetre_debut = debut - pd.DateOffset(days=MARGE_JOURS)
    fenetre_fin = fin + pd.DateOffset(days=MARGE_JOURS)

    serie = conso_jour.loc[fenetre_debut:fenetre_fin]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(serie.index, serie.values, color="steelblue", linewidth=1.3)
    ax.axvspan(debut, fin, color="red", alpha=0.15, label="Confinement (dates officielles)")
    ax.axvline(debut, color="red", linestyle="--", linewidth=1)
    ax.axvline(fin, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"{nom} : conso journalière moyenne (MW) — {debut.date()} à {fin.date()}\n"
                 f"(+/- {MARGE_JOURS} jours de référence)")
    ax.set_ylabel("Consommation moyenne (MW)")
    ax.legend()
    fig.tight_layout()

    chemin_fig = FIG_DIR / f"covid_zoom_{nom.replace(' ', '_').lower()}.png"
    fig.savefig(chemin_fig, dpi=120)
    plt.close(fig)
    print(f"  → {chemin_fig}")

    avant = conso_jour.loc[fenetre_debut:debut - pd.DateOffset(days=1)].mean()
    pendant = conso_jour.loc[debut:fin].mean()
    apres = conso_jour.loc[fin + pd.DateOffset(days=1):fenetre_fin].mean()
    print(f"    Moyenne avant  : {avant:,.0f} MW")
    print(f"    Moyenne pendant: {pendant:,.0f} MW  ({(pendant/avant - 1)*100:+.1f}% vs avant)")
    print(f"    Moyenne après  : {apres:,.0f} MW")


def vue_ensemble(conso_jour):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(conso_jour.index, conso_jour.values, color="steelblue", linewidth=0.8)
    for nom, debut, fin in PERIODES_COVID:
        ax.axvspan(pd.Timestamp(debut, tz="Europe/Paris"),
                   pd.Timestamp(fin, tz="Europe/Paris"), color="red", alpha=0.2)
    ax.set_title("Conso journalière moyenne 2016-2026, 3 confinements surlignés")
    ax.set_ylabel("Consommation moyenne (MW)")
    fig.tight_layout()
    chemin_fig = FIG_DIR / "covid_vue_ensemble.png"
    fig.savefig(chemin_fig, dpi=120)
    plt.close(fig)
    print(f"\n  → {chemin_fig}")


if __name__ == "__main__":
    conso_jour = charger()

    print("Zoom sur chaque confinement (dates officielles vs données réelles) :\n")
    for nom, debut, fin in PERIODES_COVID:
        print(f"{nom} ({debut} → {fin})")
        zoom_periode(conso_jour, nom, debut, fin)
        print()

    vue_ensemble(conso_jour)

 
