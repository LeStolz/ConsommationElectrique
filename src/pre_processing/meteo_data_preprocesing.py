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
7. sauvegarde le résultat dans data/processed/.

Important (à comprendre, pas juste à exécuter) :
Ce script construit la table d'observations météo *brute et complète*.
L'interpolation faite ici peut utiliser des points avant ET après un trou
(c'est une opération de nettoyage de données, pas encore une prévision).
La règle "pas de donnée après 14h le jour J" s'appliquera plus tard, au
moment de construire les variables (features) pour chaque prévision,
pas ici. Les deux étapes sont volontairement séparées.

Les 13 stations couvrent maintenant les 13 régions de France métropolitaine
(une station par région, vérifié contre la liste officielle des postes
SYNOP de Météo-France). La moyenne pondérée par population reste malgré
tout une approximation : une seule station par région masque les écarts
météo à l'intérieur d'une même région (ex. littoral vs intérieur des
terres), et les poids de population sont des ordres de grandeur, pas des
chiffres précis par bassin de consommation électrique.
"""

import pandas as pd
from pathlib import Path


# ============================================================
# PARAMÈTRES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "synop_meteo"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ANNEES = range(2016, 2027)  # à ajuster selon les fichiers que tu as téléchargés
# ATTENTION : la période cible du projet a été étendue à 2016 -> sept. 2026.
# Il faut donc re-télécharger/compléter les fichiers synop_2016.csv,
# synop_2017.csv, synop_2018.csv (anciens, absents jusqu'ici) et
# synop_2026.csv (données de l'année en cours) dans data/raw/synop_meteo/,
# sinon ce script les ignorera silencieusement ([absent] affiché) et la
# météo manquera sur ces années lors de la fusion.

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

# Population (ordre de grandeur, source INSEE) de la région représentée par
# chaque station, utilisée comme poids pour la moyenne pondérée. Ce ne sont
# pas des chiffres au-ha habitant près, c'est une pondération volontairement
# approximative (pas besoin de plus de précision pour ce qu'on en fait).
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

# Colonnes du fichier brut qu'on garde (voir le dictionnaire SYNOP pour le reste)
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


# ============================================================
# LECTURE ET FILTRAGE D'UNE ANNÉE
# ============================================================

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


# ============================================================
# NETTOYAGE ET CONVERSION DES UNITÉS
# ============================================================

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


# ============================================================
# PASSAGE À L'HEURE + AGRÉGATION NATIONALE (simple ET pondérée)
# ============================================================

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

        sous_df["geo_id_wmo"] = station_id
        series_par_station.append(sous_df)

    # Toutes les stations empilées, avec leur identifiant conservé (nécessaire
    # pour pondérer par population ensuite).
    concat = pd.concat(series_par_station)
    concat["poids"] = concat["geo_id_wmo"].map(POPULATION)

    # --- 1. Moyenne simple (comme avant) : chaque station compte pareil ---
    national = concat.groupby(concat.index)[variables].mean()

    # --- 2. Moyenne pondérée par population régionale ---
    # Pour chaque variable, on ignore les stations sans valeur cette
    # heure-là à la fois dans la somme pondérée ET dans la somme des poids
    # (sinon une station manquante fausserait le dénominateur).
    colonnes_ponderees = {}
    for var in variables:
        poids_valides = concat["poids"].where(concat[var].notna())
        valeur_x_poids = (concat[var] * poids_valides).fillna(0)
        somme_ponderee = valeur_x_poids.groupby(concat.index).sum()
        somme_poids = poids_valides.fillna(0).groupby(concat.index).sum()
        colonnes_ponderees[f"{var}_pondere_pop"] = somme_ponderee / somme_poids.replace(0, pd.NA)

    national_pondere = pd.DataFrame(colonnes_ponderees)

    national = national.join(national_pondere)
    national = national.reset_index().rename(columns={"validity_time": "timestamp_utc"})

    national["timestamp_paris"] = national["timestamp_utc"].dt.tz_convert("Europe/Paris")
    colonnes = ["timestamp_utc", "timestamp_paris"] + [
        c for c in national.columns if c not in ("timestamp_utc", "timestamp_paris")
    ]
    national = national[colonnes]

    for col in ("precip_1h_mm", "precip_1h_mm_pondere_pop"):
        n_negatives = (national[col] < 0).sum()
        if n_negatives > 0:
            print(f"  Anomalie détectée sur {col} : {n_negatives} valeurs de "
                  f"précipitation négatives après interpolation (artefact "
                  f"numérique) -> tronquées à 0")
        national[col] = national[col].clip(lower=0)

    return national


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

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