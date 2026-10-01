"""
Fusion des trois tables prétraitées en un seul jeu de données.

Ce script fusionne :
- consommation_electricite_horaire.csv (la cible)
- synop_national_horaire.csv (météo)
- variables_calendaires_horaire.csv (calendrier)

sur la clé commune timestamp_utc, et sauvegarde le résultat dans
data/processed/dataset_final.csv.

Choix de jointure  :
On fait une jointure "left" sur la table de CONSOMMATION, car c'est elle
qui définit la cible à prévoir : on ne veut jamais perdre une heure de
conso observée simplement parce que la météo ou le calendrier manquerait
pour cette heure-là. Les trous résultants (s'il y en a) sont documentés,
pas comblés silencieusement.
"""

import pandas as pd
from pathlib import Path


# PARAMS

BASE_DIR = Path(__file__).resolve().parent.parent.parent
 
PROCESSED_DIR = BASE_DIR / "data" / "processed"
 
FICHIER_CONSO = PROCESSED_DIR / "consommation_electricite_horaire.csv"
FICHIER_METEO = PROCESSED_DIR / "synop_national_horaire.csv"
FICHIER_CALENDRIER = PROCESSED_DIR / "variables_calendaires_horaire.csv"
 
FICHIER_SORTIE = PROCESSED_DIR / "dataset_final.csv"


def lire_table(chemin, nom):
    if not chemin.exists():
        raise SystemExit(f"Fichier introuvable : {chemin}")
    df = pd.read_csv(chemin, parse_dates=["timestamp_utc"])
    print(f"{nom:12s} : {df.shape[0]:>7} lignes   "
          f"{df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}")
    return df


def fusionner(conso, meteo, calendrier):

    print("\nFusion sur timestamp_utc (jointure left sur la conso)...")

   
    meteo = meteo.drop(columns=["timestamp_paris"])
    calendrier = calendrier.drop(columns=["timestamp_paris"])

    colonnes_redondantes = ["heure", "jour_semaine", "mois"]
    assert (conso[colonnes_redondantes].values ==
            calendrier[colonnes_redondantes].values).all(), (
        "Les variables calendaires calculées dans les deux scripts ne "
        "correspondent pas : à investiguer avant de continuer."
    )
    calendrier = calendrier.drop(columns=colonnes_redondantes)

    df = conso.merge(meteo, on="timestamp_utc", how="left")
    df = df.merge(calendrier, on="timestamp_utc", how="left")

    print(f"Table fusionnée : {df.shape}")

    return df




def verifier_trous(df):

    print("\n Vérification des trous introduits par la fusion ")

    colonnes_meteo = ["temperature_c", "humidite_pct", "vent_vitesse_ms"]
    colonnes_calendrier = ["heure", "jour_semaine", "ferie", "vacances"]

    for col in colonnes_meteo + colonnes_calendrier:
        if col not in df.columns:
            continue
        n_manquants = df[col].isna().sum()
        if n_manquants > 0:
            print(f"  {col:20s} : {n_manquants} valeurs manquantes après fusion")
            premieres = df.loc[df[col].isna(), "timestamp_utc"].head(3).tolist()
            dernieres = df.loc[df[col].isna(), "timestamp_utc"].tail(3).tolist()
            print(f"    exemples : {premieres} ... {dernieres}")

    print()
    print("Doublons de timestamp_utc :", df["timestamp_utc"].duplicated().sum())



if __name__ == "__main__":

    conso = lire_table(FICHIER_CONSO, "Conso")
    meteo = lire_table(FICHIER_METEO, "Météo")
    calendrier = lire_table(FICHIER_CALENDRIER, "Calendrier")

    df = fusionner(conso, meteo, calendrier)

    verifier_trous(df)

    print("\nAperçu :")
    print(df.head())
    print()
    print("Colonnes finales :", df.columns.tolist())

    df.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")
    print(f"\nSauvegardé : {FICHIER_SORTIE}")