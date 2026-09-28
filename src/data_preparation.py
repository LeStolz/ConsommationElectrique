import requests
import pandas as pd
from pathlib import Path


# ============================================================
# PARAMÈTRES
# ============================================================

DATASET = "eco2mix-national-cons-def"

API_URL = (
    f"https://odre.opendatasoft.com/api/explore/v2.1/"
    f"catalog/datasets/{DATASET}/records"
)

DATE_DEBUT = "2019-01-01"
DATE_FIN = "2026-01-01"

OUTPUT_DIR = Path("data/raw/rte")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXTRACTION
# ============================================================

def recuperer_donnees():

    print("Téléchargement des données RTE...")
    print("Période : 2019-2025\n")

    toutes_les_lignes = []

    mois = pd.date_range(
        start=DATE_DEBUT,
        end="2025-12-01",
        freq="MS"
    )

    for date_debut_mois in mois:

        date_fin_mois = date_debut_mois + pd.offsets.MonthBegin(1)

        debut = date_debut_mois.strftime("%Y-%m-%dT%H:%M:%S")
        fin = date_fin_mois.strftime("%Y-%m-%dT%H:%M:%S")

        print(
            f"Téléchargement : "
            f"{date_debut_mois.strftime('%Y-%m')}..."
        )

        params = {
            "where": (
                f"date_heure >= '{debut}' "
                f"AND date_heure < '{fin}'"
            ),
            "limit": 100,
            "offset": 0
        }

        lignes_mois = []

        while True:

            response = requests.get(
                API_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            records = data.get("results", [])

            if not records:
                break

            lignes_mois.extend(records)

            if len(records) < params["limit"]:
                break

            params["offset"] += params["limit"]

        toutes_les_lignes.extend(lignes_mois)

        print(
            f"  → {len(lignes_mois)} lignes récupérées"
        )

    df = pd.DataFrame(toutes_les_lignes)

    print(
        f"\nTotal : {len(df)} lignes récupérées."
    )

    return df


# ============================================================
# NETTOYAGE
# ============================================================

def nettoyer_donnees(df):

    print("\nNettoyage des données...")

    # Conversion de la date
    df["date_heure"] = pd.to_datetime(
        df["date_heure"],
        errors="coerce"
    )

    # Conversion de la consommation
    df["consommation"] = pd.to_numeric(
        df["consommation"],
        errors="coerce"
    )

    # Garder uniquement les colonnes utiles
    df = df[
        [
            "date_heure",
            "consommation",
            "nature"
        ]
    ]

    # Supprimer les dates ou consommations manquantes
    df = df.dropna(
        subset=[
            "date_heure",
            "consommation"
        ]
    )

    # Trier chronologiquement
    df = df.sort_values("date_heure")

    # Supprimer d'éventuels doublons
    df = df.drop_duplicates(
        subset=["date_heure"]
    )

    return df


# ============================================================
# CONVERSION 30 MIN → HORAIRE
# ============================================================

def convertir_en_horaire(df):

    print("\nConversion 30 minutes → 1 heure...")

    df = df.set_index("date_heure")

    df_horaire = (
        df["consommation"]
        .resample("1h")
        .mean()
        .reset_index()
    )

    df_horaire = df_horaire.rename(
        columns={
            "consommation": "consommation_mw"
        }
    )

    return df_horaire


# ============================================================
# SAUVEGARDE
# ============================================================

def sauvegarder(df, nom_fichier):

    chemin = OUTPUT_DIR / nom_fichier

    df.to_csv(
        chemin,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Données sauvegardées : {chemin}"
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # 1. Télécharger
    df = recuperer_donnees()

    print(
        "\nDimensions initiales :",
        df.shape
    )

    # 2. Nettoyer
    df = nettoyer_donnees(df)

    print(
        "Après nettoyage :",
        df.shape
    )

    # 3. Sauvegarder les données brutes nettoyées
    sauvegarder(
        df,
        "rte_consommation_30min_2019_2025.csv"
    )

    # 4. Passer en horaire
    df_horaire = convertir_en_horaire(df)

    print(
        "Données horaires :",
        df_horaire.shape
    )

    # 5. Sauvegarder les données horaires
    sauvegarder(
        df_horaire,
        "rte_consommation_horaire_2019_2025.csv"
    )

    print(
        "\nExtraction terminée !"
    )