"""
Construction des variables calendaires.

Ce script construit, pour chaque heure de la période étudiée :
- l'heure, le jour de la semaine, le mois (en heure de Paris, car ce sont
  des notions de calendrier civil, pas des instants UTC) ;
- le week-end
- les jours fériés (booléen `ferie` -- le nom du jour férié n'est pas
  gardé dans la sortie finale, pas utile pour filtrer/modéliser)
- les vacances scolaires

"""

import pandas as pd
import holidays
from pathlib import Path


# PARAMS

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "calendrier"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


FICHIER_VACANCES_BRUT = RAW_DIR / "fr-en-calendrier-scolaire.csv"


FICHIER_VACANCES_COMPLEMENT = RAW_DIR / "vacances_scolaires_2016_2017_complement.csv"

DATE_DEBUT = "2016-01-01"
DATE_FIN = "2026-10-01"

ZONES = ["Zone A", "Zone B", "Zone C"]

ALIAS_COLONNES = {
    "date_debut": ["Date de début", "start_date", "Date début"],
    "date_fin": ["Date de fin", "end_date", "Date fin"],
    "zones": ["Zones", "zones", "Zone"],
    "population": ["Population", "population"],
}


def construire_jours_feries():
    print("Calcul des jours fériés...")
    annees = range(2016, 2027)
    fr = holidays.France(years=list(annees))
    df = pd.DataFrame(
        [(pd.Timestamp(d), nom) for d, nom in fr.items()],
        columns=["date", "nom_ferie"]
    )
    print(f"  → {len(df)} jours fériés calculés")
    return df



def _trouver_colonne(df, candidats, nom_logique):
    for c in candidats:
        if c in df.columns:
            return c
    raise SystemExit(
        f"Impossible de trouver la colonne '{nom_logique}'.\n"
        f"Colonnes disponibles dans le fichier : {df.columns.tolist()}\n"
        f"Ajoute le nom exact à ALIAS_COLONNES['{nom_logique}'] et relance."
    )


def lire_vacances_scolaires_brut():
    print("Lecture du fichier vacances scolaires...")

    if not FICHIER_VACANCES_BRUT.exists():
        raise SystemExit(
            f"Fichier introuvable : {FICHIER_VACANCES_BRUT}\n"
            "Télécharge-le depuis :\n"
            "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
            "fr-en-calendrier-scolaire/exports/csv?use_labels=true\n"
            f"et place-le dans {RAW_DIR}/"
        )

    df = pd.read_csv(FICHIER_VACANCES_BRUT, sep=";")

    print(f"  → {len(df)} lignes lues, colonnes : {df.columns.tolist()}")

    if FICHIER_VACANCES_COMPLEMENT.exists():
        print(f"\nLecture du complément 2016-2017 (trou du dataset officiel)...")
        df_complement = pd.read_csv(FICHIER_VACANCES_COMPLEMENT, sep=";")
        print(f"  → {len(df_complement)} lignes ajoutées (source : arrêté du "
              f"21 janvier 2014, Légifrance)")
        df = pd.concat([df, df_complement], ignore_index=True)
    else:
        print(f"\nATTENTION : {FICHIER_VACANCES_COMPLEMENT.name} introuvable "
              f"-> le trou Zone A/B/C de janvier 2016 à août 2017 ne sera "
              f"PAS comblé (vacances=0 sur cette période, même si en vrai "
              f"il y a des vacances).")

    return df


def construire_calendrier_vacances(df_vacances):

    print("\nConstruction du calendrier journalier des vacances...")

    col_debut = _trouver_colonne(df_vacances, ALIAS_COLONNES["date_debut"], "date_debut")
    col_fin = _trouver_colonne(df_vacances, ALIAS_COLONNES["date_fin"], "date_fin")
    col_zones = _trouver_colonne(df_vacances, ALIAS_COLONNES["zones"], "zones")
    col_population = _trouver_colonne(df_vacances, ALIAS_COLONNES["population"], "population")

    df = df_vacances.copy()

    masque_population = (
        df[col_population].astype(str).str.strip().eq("-")
        | df[col_population].astype(str).str.contains("lève", case=False, na=False)
    )
    df = df[masque_population]
    df = df[df[col_zones].isin(ZONES)]

    df[col_debut] = pd.to_datetime(df[col_debut], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    df[col_fin] = pd.to_datetime(df[col_fin], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    df = df.dropna(subset=[col_debut, col_fin])

    jours = pd.date_range(DATE_DEBUT, DATE_FIN, freq="D")


    nb_zones_vacances = pd.Series(0, index=jours)

    for zone in ZONES:
        en_vacances_zone = pd.Series(False, index=jours)
        for _, ligne in df[df[col_zones] == zone].iterrows():
            masque = (jours >= ligne[col_debut]) & (jours <= ligne[col_fin])
            en_vacances_zone |= masque
        nb_zones_vacances += en_vacances_zone.astype(int)

    resultat = pd.DataFrame({"date": jours, "vacances": nb_zones_vacances.values})
    print(f"  → répartition du nombre de zones en vacances par jour :")
    print(resultat["vacances"].value_counts().sort_index())

    return resultat



def construire_table_horaire(df_feries, df_vacances):

    index_utc = pd.date_range(
        DATE_DEBUT, DATE_FIN, freq="1h", tz="UTC", inclusive="left"
    )

    df = pd.DataFrame({"timestamp_utc": index_utc})
    df["timestamp_paris"] = df["timestamp_utc"].dt.tz_convert("Europe/Paris")

    df["heure"] = df["timestamp_paris"].dt.hour
    df["jour_semaine"] = df["timestamp_paris"].dt.dayofweek
    df["mois"] = df["timestamp_paris"].dt.month
    df["weekend"] = df["jour_semaine"].isin([5, 6])

    df["date"] = df["timestamp_paris"].dt.normalize().dt.tz_localize(None)

    df = df.merge(df_feries, on="date", how="left")
    df["ferie"] = df["nom_ferie"].notna()
  
    df = df.drop(columns=["nom_ferie"])

    df = df.merge(df_vacances, on="date", how="left")
    df["vacances"] = df["vacances"].fillna(0).astype(int)

    df = df.drop(columns=["date"])
    df = df[["timestamp_utc", "timestamp_paris", "heure", "jour_semaine", "mois",
              "weekend", "ferie", "vacances"]]

    return df



if __name__ == "__main__":

    df_feries = construire_jours_feries()

    df_vacances_brut = lire_vacances_scolaires_brut()
    df_vacances = construire_calendrier_vacances(df_vacances_brut)

    df_final = construire_table_horaire(df_feries, df_vacances)

    print("\nTable finale :", df_final.shape)
    print(df_final.head())
    print()
    print("Répartition :")
    print("  Week-end                      :", df_final["weekend"].sum())
    print("  Fériés                        :", df_final["ferie"].sum())
    print("  Heures avec >= 1 zone vacances :", (df_final["vacances"] > 0).sum())
    print("  Répartition du nombre de zones en vacances (par heure) :")
    print(df_final["vacances"].value_counts().sort_index())

    chemin_sortie = PROCESSED_DIR / "variables_calendaires_horaire.csv"
    df_final.to_csv(chemin_sortie, index=False, encoding="utf-8")
    print(f"\nSauvegardé : {chemin_sortie}")