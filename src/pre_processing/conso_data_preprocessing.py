"""
Prétraitement des données de consommation électrique (RTE / éCO2mix).

Ce script :
1. lit le fichier horaire brut déjà téléchargé (rte_consommation_horaire...csv) ;
2. convertit explicitement l'horodatage en UTC (pas d'hypothèse implicite) ;
3. construit une colonne timestamp_paris en plus de timestamp_utc, pour
   garder une trace des deux représentations ;
4. dérive les variables calendaires de base (heure, jour de semaine, mois,
   année, saison) à partir de l'heure de Paris, pas de l'UTC, car ce sont
   des notions de calendrier civil (voir discussion précédente) ;
5. remplace la colonne "corona" par une colonne "covid19", avec la période
   exacte documentée plutôt qu'une simple étiquette Oui/Non ;
6. gère les valeurs manquantes de consommation.

Pourquoi deux colonnes de timestamp :
timestamp_utc sert de clé de fusion avec les tables météo et calendrier
(toutes en UTC). timestamp_paris sert à vérifier/auditer les variables
calendaires sans avoir à reconvertir à chaque fois. Les deux sont gardées
pour la traçabilité, comme demandé.
"""

import pandas as pd
from pathlib import Path


# PARAmS

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "rte_consommation"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True) 


FICHIER_BRUT = RAW_DIR / "rte_consommation_horaire_2019_2025.csv"


COVID_DEBUT = "2020-03-17"
COVID_FIN = "2021-06-30"



def lire_et_convertir():

    print("Lecture du fichier conso horaire brut...")

    if not FICHIER_BRUT.exists():
        raise SystemExit(f"Fichier introuvable : {FICHIER_BRUT}")

    df = pd.read_csv(FICHIER_BRUT)
    print(f"  → {len(df)} lignes lues")


    df["timestamp_utc"] = pd.to_datetime(df["date_heure"], errors="coerce", utc=True)

    n_dates_invalides = df["timestamp_utc"].isna().sum()
    if n_dates_invalides > 0:
        print(f"  ATTENTION : {n_dates_invalides} horodatages n'ont pas pu être "
              f"convertis et seront supprimés.")
        df = df.dropna(subset=["timestamp_utc"])

   
    df["timestamp_paris"] = df["timestamp_utc"].dt.tz_convert("Europe/Paris")

    df = df.rename(columns={"consommation_mw": "consommation_mw"})  

    return df[["timestamp_utc", "timestamp_paris", "consommation_mw"]]


# 
#Traitement des VALEURS MANQUANTES

def gerer_valeurs_manquantes(df):

    print("\nGestion des valeurs manquantes...")

    n_manquantes = df["consommation_mw"].isna().sum()
    print(f"  → {n_manquantes} valeurs manquantes trouvées")

    if n_manquantes > 0:
        df = df.sort_values("timestamp_utc")
      
        df["consommation_mw"] = df["consommation_mw"].interpolate(
            method="linear", limit=3
        )
        n_restantes = df["consommation_mw"].isna().sum()
        if n_restantes > 0:
            print(f"  ATTENTION : {n_restantes} valeurs restent manquantes "
                  f"(trou > 3h), à examiner manuellement.")

    return df


# VARIABLES CALENDAIRES DE BASE + COVID

def ajouter_variables(df):

    print("\nAjout des variables calendaires et de la période Covid...")

    df["heure"] = df["timestamp_paris"].dt.hour
    df["jour_semaine"] = df["timestamp_paris"].dt.dayofweek  # 0=lundi
    df["mois"] = df["timestamp_paris"].dt.month
    df["annee"] = df["timestamp_paris"].dt.year

    saisons = {
        12: "Hiver", 1: "Hiver", 2: "Hiver",
        3: "Printemps", 4: "Printemps", 5: "Printemps",
        6: "Été", 7: "Été", 8: "Été",
        9: "Automne", 10: "Automne", 11: "Automne",
    }
    df["saison"] = df["mois"].map(saisons)

    date_paris = df["timestamp_paris"].dt.tz_localize(None)
    df["covid19"] = date_paris.between(COVID_DEBUT, COVID_FIN)

    print(f"  → {df['covid19'].sum()} heures classées en période Covid "
          f"({COVID_DEBUT} au {COVID_FIN})")

    return df

if __name__ == "__main__":

    df = lire_et_convertir()
    df = gerer_valeurs_manquantes(df)
    df = ajouter_variables(df)

    print("\nTable finale :", df.shape)
    print(df.head())

    print("\n--- Vérification rapide ---")
    print("Doublons de timestamp_utc :", df["timestamp_utc"].duplicated().sum())
    print("Valeurs manquantes restantes :", df["consommation_mw"].isna().sum())
    print("Plage :", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())

    chemin_sortie = PROCESSED_DIR / "consommation_electricite_horaire.csv"
    df.to_csv(chemin_sortie, index=False, encoding="utf-8")
    print(f"\nSauvegardé : {chemin_sortie}")