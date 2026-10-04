"""
Prétraitement des données météo SYNOP (Météo-France).

Ce script :
1. lit les fichiers bruts synop_AAAA.csv (un par année) ;
2. filtre sur un petit nombre de stations représentatives de la métropole ;
3. garde uniquement les variables météo utiles à la prévision de conso ;
4. convertit la température en degrés Celsius ;
5. ramène les observations (souvent toutes les 3h) à une fréquence horaire ;
6. agrège les stations en une série météo "nationale", avec DEUX variantes :
   - une moyenne simple (chaque station compte pareil) ;
   - une moyenne pondérée par la population de la région représentée par
     chaque station (colonnes suffixées "_pondere_pop") -> plus pertinente
     pour la conso électrique, qui suit surtout les zones peuplées
     (Paris pèse beaucoup plus que Rennes dans la conso nationale).

"""

import pandas as pd
from pathlib import Path


# PARAMS

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "synop_meteo"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATE_DEBUT = "2016-01-01"
DATE_FIN = "2026-10-01"

ANNEES = range(2016, 2027)

# Stations retenues : une par région de France métropolitaine (13 régions
# couvertes sur 13, vérifié contre la liste officielle des postes SYNOP).
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
    7240: "TOURS",             # Centre-Val de Loire
    7280: "DIJON-LONGVIC",     # Bourgogne-Franche-Comté
    7027: "CAEN-CARPIQUET",    # Normandie
    7761: "AJACCIO",           # Corse
}

# Population (ordre de grandeur, source INSEE)
POPULATION = {
    7190: 5_500_000,   # Strasbourg - Grand Est
    7481: 8_100_000,   # Lyon - Auvergne-Rhône-Alpes
    7015: 6_000_000,   # Lille - Hauts-de-France
    7149: 12_300_000,  # Orly - Île-de-France
    7650: 5_100_000,   # Marignane - Provence-Alpes-Côte d'Azur
    7510: 6_100_000,   # Bordeaux - Nouvelle-Aquitaine
    7222: 3_800_000,   # Nantes - Pays de la Loire
    7130: 3_400_000,   # Rennes - Bretagne
    7630: 6_100_000,   # Toulouse - Occitanie
    7240: 2_600_000,   # Tours - Centre-Val de Loire
    7280: 2_800_000,   # Dijon - Bourgogne-Franche-Comté
    7027: 3_300_000,   # Caen - Normandie
    7761: 340_000,     # Ajaccio - Corse
}

# Colonnes du fichier brut qu'on garde 
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
        # Les stations ne publient souvent qu'une observation toutes les
        # 3h -> on reporte la dernière valeur connue (ffill) sur les
        # heures intermédiaires manquantes, limité à 3h. Pas
        # d'interpolation linéaire : on ne veut jamais qu'une valeur
        # dépende d'une observation future par rapport à son propre
        # horodatage (ex. météo[13h] = météo[14h] = dernière valeur
        # connue à 12h, pas une moyenne entre 12h et 15h).
        sous_df = sous_df.ffill(limit=3)

        sous_df["geo_id_wmo"] = station_id
        series_par_station.append(sous_df)

    concat = pd.concat(series_par_station)
    concat["poids"] = concat["geo_id_wmo"].map(POPULATION)

    # 1. Moyenne simple 
    national = concat.groupby(concat.index)[variables].mean()

    #  2. Moyenne pondérée par population régionale
    # Pour chaque variable, on ignore les stations sans valeur cette
    # heure-là à la fois dans la somme pondérée ET dans la somme des poids
    colonnes_ponderees = {}
    for var in variables:
        poids_valides = concat["poids"].where(concat[var].notna())
        valeur_x_poids = (concat[var] * poids_valides).fillna(0)
        somme_ponderee = valeur_x_poids.groupby(concat.index).sum()
        somme_poids = poids_valides.fillna(0).groupby(concat.index).sum()
        colonnes_ponderees[f"{var}_pondere_pop"] = somme_ponderee / somme_poids.replace(0, pd.NA)

    national_pondere = pd.DataFrame(colonnes_ponderees)

    national = national.join(national_pondere)

    toutes_colonnes = variables + [f"{v}_pondere_pop" for v in variables]

    grille_cible = pd.date_range(DATE_DEBUT, DATE_FIN, freq="1h", tz="UTC", inclusive="left")
    national = national.reindex(grille_cible)

    n_absentes = national[toutes_colonnes].isna().all(axis=1).sum()
    print(f"  → {n_absentes} heures où les 13 stations sont absentes en même temps")

    n_avant = national[toutes_colonnes].isna().sum().sum()
    national[toutes_colonnes] = national[toutes_colonnes].ffill(limit=3)
    n_apres_report = national[toutes_colonnes].isna().sum().sum()
    print(f"  Étape 2 (report <=3h, valeur précédente) : "
          f"{n_avant - n_apres_report} valeurs comblées, "
          f"{n_apres_report} valeurs manquantes restantes")


    national["_heure_du_jour"] = national.index.hour
    for h in range(24):
        masque = national["_heure_du_jour"] == h
        national.loc[masque, toutes_colonnes] = (
            national.loc[masque, toutes_colonnes].sort_index().ffill()
        )
    national = national.drop(columns=["_heure_du_jour"])

    n_apres_fallback = national[toutes_colonnes].isna().sum().sum()
    print(f"  Étape 3 (même heure, jour précédent dispo) : "
          f"{n_apres_report - n_apres_fallback} valeurs comblées en plus, "
          f"{n_apres_fallback} valeurs manquantes restantes au total "
          f"(uniquement possible tout début de série, si aucun jour "
          f"précédent n'existe encore).")

    national = national.reset_index().rename(columns={"index": "timestamp_utc"})

    national["timestamp_paris"] = national["timestamp_utc"].dt.tz_convert("Europe/Paris")
    colonnes = ["timestamp_utc", "timestamp_paris"] + [
        c for c in national.columns if c not in ("timestamp_utc", "timestamp_paris")
    ]
    national = national[colonnes]

    for col in ("precip_1h_mm", "precip_1h_mm_pondere_pop"):
        n_negatives = (national[col] < 0).sum()
        if n_negatives > 0:
            print(f"  Anomalie détectée sur {col} : {n_negatives} valeurs de "
                  f"précipitation négatives -> tronquées à 0")
        national[col] = national[col].clip(lower=0)

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
    print("Température (moyenne simple) min / max :",
          df_national["temperature_c"].min(), "/", df_national["temperature_c"].max())
    print("Température (moyenne pondérée pop) min / max :",
          df_national["temperature_c_pondere_pop"].min(), "/",
          df_national["temperature_c_pondere_pop"].max())
    ecart_moyen = (df_national["temperature_c_pondere_pop"] - df_national["temperature_c"]).mean()
    print(f"Écart moyen (pondérée - simple) sur la température : {ecart_moyen:.3f} °C")