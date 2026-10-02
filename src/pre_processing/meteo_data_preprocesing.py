"""
Prétraitement des données météo SYNOP (Météo-France).

Ce script :
1. lit les fichiers bruts synop_AAAA.csv (un par année) ;
2. filtre sur un petit nombre de stations représentatives de la métropole ;
3. garde uniquement les variables météo utiles à la prévision de conso ;
4. convertit la température en degrés Celsius ;
5. ramène les observations (souvent toutes les 3h) à une fréquence horaire ;
6. agrège les stations en une seule série météo "nationale" (moyenne) ;
7. sauvegarde le résultat dans data/processed/.

"""

import pandas as pd
from pathlib import Path


# 
# PARAMS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "synop_meteo"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ANNEES = range(2016, 2027)
 
# Stations retenues : couverture nord/sud/est/ouest/centre + grandes métropoles
STATIONS = {
    7190: "STRASBOURG-ENTZHEIM",
    7481: "LYON-ST EXUPERY",
    7015: "LILLE-LESQUIN",
    7149: "ORLY",
    7650: "MARIGNANE",
    7510: "BORDEAUX-MERIGNAC",
    7222: "NANTES-BOUGUENAIS",
    7130: "RENNES-ST JACQUES",
    7630: "TOULOUSE-BLAGNAC",
}
 
# Colonnes du fichier brut qu'on garde ( le dictionnaire SYNOP)
COLONNES_UTILES = [
    "geo_id_wmo",   # identifiant de la station
    "validity_time",  # horodatage de l'observation (déjà en UTC)
    "t",    # température (en Kelvin dans le fichier brut)
    "td",   # température du point de rosée (Kelvin)
    "u",    # humidité relative (%)
    "dd",   # direction du vent (degrés)
    "ff",   # vitesse du vent (m/s)
    "n",    # nébulosité totale
    "rr1",  # précipitations sur la dernière heure (mm)
]
 
 

 
def lire_annee(annee):
    """Lit un fichier synop_AAAA.csv et ne garde que les stations retenues."""
 
    chemin = RAW_DIR / f"synop_{annee}.csv"
 
    if not chemin.exists():
        print(f"  [absent] {chemin}")
        return None
 
    print(f"Lecture {annee}...")
 
    df = pd.read_csv(
        chemin,
        sep=";",
        usecols=lambda col: col in COLONNES_UTILES,
        na_values=["", "mq"], 
        low_memory=False,
    )
 
    
    df["geo_id_wmo"] = pd.to_numeric(df["geo_id_wmo"], errors="coerce")
 
    df = df[df["geo_id_wmo"].isin(STATIONS.keys())].copy()
 
    print(f"  → {len(df)} lignes pour les {len(STATIONS)} stations retenues")
 
    return df
 
 

 
def nettoyer(df):
 
    print("\nNettoyage et conversion des unités...")
 
    df["validity_time"] = pd.to_datetime(df["validity_time"], utc=True, errors="coerce")
 
    # Temp: Kelvin -> Celsius
    df["temperature_c"] = df["t"] - 273.15
    df["temperature_point_rosee_c"] = df["td"] - 273.15
 
    df = df.rename(columns={
        "u": "humidite_pct",
        "dd": "vent_direction_deg",
        "ff": "vent_vitesse_ms",
        "n": "nebulosite",
        "rr1": "precip_1h_mm",
    })
 
    colonnes_finales = [
        "geo_id_wmo", "validity_time",
        "temperature_c", "temperature_point_rosee_c",
        "humidite_pct", "vent_direction_deg", "vent_vitesse_ms",
        "nebulosite", "precip_1h_mm",
    ]
    df = df[colonnes_finales]
 
    
    df = df.dropna(subset=["validity_time", "temperature_c"])
 
    # Supprimer les doublons (même station, même instant)
    df = df.drop_duplicates(subset=["geo_id_wmo", "validity_time"])
 
    return df
 
 

 
def mettre_a_heure_et_agreger(df):
   
 
    print("\nPassage à l'heure et agrégation des stations...")
 
    variables = [
        "temperature_c", "temperature_point_rosee_c",
        "humidite_pct", "vent_direction_deg", "vent_vitesse_ms",
        "nebulosite", "precip_1h_mm",
    ]
 
    series_par_station = []
 
    for station_id in STATIONS:
        sous_df = df[df["geo_id_wmo"] == station_id].set_index("validity_time")
        sous_df = sous_df[variables].sort_index()
 
        sous_df = sous_df.resample("1h").mean()
        sous_df = sous_df.interpolate(method="linear", limit=3)
 
        series_par_station.append(sous_df)
 
    # Moyenne des stations à chaque heure (= "météo nationale")
    concat = pd.concat(series_par_station)
    national = concat.groupby(concat.index).mean()
    national = national.reset_index().rename(columns={"validity_time": "timestamp_utc"})
 
    
    national["timestamp_paris"] = national["timestamp_utc"].dt.tz_convert("Europe/Paris")
    colonnes = ["timestamp_utc", "timestamp_paris"] + [
        c for c in national.columns if c not in ("timestamp_utc", "timestamp_paris")
    ]
    national = national[colonnes]
 
  
    n_negatives = (national["precip_1h_mm"] < 0).sum()
    if n_negatives > 0:
        print(f"  Anomalie détectée : {n_negatives} valeurs de précipitation "
              f"négatives après interpolation (artefact numérique) -> tronquées à 0")
    national["precip_1h_mm"] = national["precip_1h_mm"].clip(lower=0)
 
    return national
 
 
 
if __name__ == "__main__":
 
    tables_annuelles = []
 
    for annee in ANNEES:
        df_annee = lire_annee(annee)
        if df_annee is not None:
            tables_annuelles.append(df_annee)
 
    if not tables_annuelles:
        raise SystemExit(
            "Aucun fichier trouvé. Vérifie que tes fichiers synop_AAAA.csv "
            f"sont bien dans {RAW_DIR}/"
        )
 
    df = pd.concat(tables_annuelles, ignore_index=True)
    print(f"\nTotal brut filtré : {df.shape}")
 
    df = nettoyer(df)
    print(f"Après nettoyage : {df.shape}")
 
    df_national = mettre_a_heure_et_agreger(df)
    print(f"Table horaire nationale : {df_national.shape}")
 
    chemin_sortie = PROCESSED_DIR / "synop_national_horaire.csv"
    df_national.to_csv(chemin_sortie, index=False, encoding="utf-8")
 
    print(f"\nSauvegardé : {chemin_sortie}")
    print(df_national.head())
 
    # Diagnostic rapide 
    print("\n--- Vérification rapide ---")
    print("Valeurs manquantes par colonne :")
    print(df_national.isna().sum())
    print("\nPrécipitations encore négatives (doit être 0) :",
          (df_national["precip_1h_mm"] < 0).sum())
    print("Température min / max :",
          df_national["temperature_c"].min(), "/", df_national["temperature_c"].max())
 
 