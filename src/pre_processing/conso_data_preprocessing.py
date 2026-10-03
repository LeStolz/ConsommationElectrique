"""
Deux fichiers bruts sont nécessaires, car aucun des deux seuls ne couvre
2016 -> septembre 2026 :

1. "Données éCO2mix nationales consolidées et définitives" (dataset
   "cons-def") sur ODRE :
   https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/export/?disjunctive.nature&q.timerange.date_heure=date_heure:%5B2015-12-31T23:00:00Z+TO+2026-09-30T21:59:59Z%5D
   -> Raison : c'est la source la plus fiable (données consolidées /
      définitives), mais elle s'arrête en réalité fin juin 2026, laissant un trou de 3 mois (juillet -> septembre 2026) que l'on doit combler avec la source "tr".
2. "Données éCO2mix nationales temps réel" (dataset "tr") sur ODRE :
   https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/export/
   -> Raison : seule source qui couvre juillet -> septembre 2026 (le trou
      laissé par "cons-def"). Ces données sont encore révisables (moins
      fiables que "cons-def"), donc on les utilise UNIQUEMENT pour combler
      ce que "cons-def" ne couvre pas, jamais pour écraser une valeur déjà
      présente dans "cons-def".

Ce script :
1. lit les deux fichiers bruts (15 minutes) ;
2. convertit l'horodatage en UTC explicitement (utc=True) ;
3. agrège chacun à l'heure (moyenne), comme pour les stations SYNOP
   ramenées à une série nationale ;
4. combine les deux : "cons-def" prioritaire, "tr" utilisé seulement pour
   les heures absentes de "cons-def" ;
5. restreint à la période cible 2016-01-01 -> 2026-09-30 ;
6. dérive les variables calendaires de base (heure, jour de semaine, mois,
   année, saison) à partir de l'heure de Paris ;
7. ajoute les variables covid19 / confinement_numero (3 périodes
   officielles distinctes, voir PERIODES_COVID) ;
8. gère les valeurs manquantes résiduelles (trous courts).
"""

import pandas as pd
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "rte_consommation"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FICHIER_DEF = RAW_DIR / "eco2mix_national_cons_def.csv"
FICHIER_TR = RAW_DIR / "eco2mix_national_tr.csv"

DATE_DEBUT = "2016-01-01"
DATE_FIN = "2026-10-01"

# Périodes de confinement national 
PERIODES_COVID = [
    ("2020-03-17", "2020-05-11"),  # confinement 1
    ("2020-10-30", "2020-12-15"),  # confinement 2
    ("2021-04-03", "2021-05-03"),  # confinement 3
]

ALIAS_COLONNES = {
    "timestamp": ["Date et Heure", "date_heure", "Date - Heure"],
    "consommation": ["Consommation (MW)", "consommation", "consommation_mw"],
    "nature": ["Nature", "nature"],
}


def _trouver_colonne(df, candidats, nom_logique):
    for c in candidats:
        if c in df.columns:
            return c
    raise SystemExit(
        f"Impossible de trouver la colonne '{nom_logique}'.\n"
        f"Colonnes disponibles : {df.columns.tolist()}\n"
        f"Ajoute le nom exact à ALIAS_COLONNES['{nom_logique}'] et relance."
    )




def lire_fichier_brut(chemin, nom_source):

    if not chemin.exists():
        raise SystemExit(
            f"Fichier introuvable : {chemin}\n"
            "Voir les instructions de téléchargement en haut de ce script."
        )

    print(f"Lecture du fichier brut {nom_source} ({chemin.name})...")

    df = pd.read_csv(chemin, sep=";", encoding="utf-8")
    print(f"  → {len(df)} lignes lues")

    col_timestamp = _trouver_colonne(df, ALIAS_COLONNES["timestamp"], "timestamp")
    col_conso = _trouver_colonne(df, ALIAS_COLONNES["consommation"], "consommation")
    col_nature = _trouver_colonne(df, ALIAS_COLONNES["nature"], "nature")

    df = df[[col_timestamp, col_conso, col_nature]].rename(columns={
        col_timestamp: "timestamp_brut",
        col_conso: "consommation_mw",
        col_nature: "source_donnee",
    })

    # conversion horaire en UTC
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_brut"], errors="coerce", utc=True)

    n_invalides = df["timestamp_utc"].isna().sum()
    if n_invalides > 0:
        print(f"  ATTENTION : {n_invalides} horodatages non convertis, supprimés.")
        df = df.dropna(subset=["timestamp_utc"])

    df["consommation_mw"] = pd.to_numeric(df["consommation_mw"], errors="coerce")

    n_vides = df["consommation_mw"].isna().sum()
    print(f"  → {n_vides} lignes avec consommation vide dans le fichier brut "
          f"(quarts d'heure pas encore consolidés, traités comme manquants)")

    return df[["timestamp_utc", "consommation_mw", "source_donnee"]].sort_values("timestamp_utc")


# AGRÉGATION 15 MIN -> HEURE


def horariser(df, nom_source):

    print(f"\nAgrégation horaire {nom_source} (moyenne des 4 quarts d'heure)...")

    df = df.copy()
    df["heure_pleine_utc"] = df["timestamp_utc"].dt.floor("h")

    agg = (
        df.groupby("heure_pleine_utc")
        .agg(
            consommation_mw=("consommation_mw", "mean"),
            source_donnee=("source_donnee", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        )
        .reset_index()
        .rename(columns={"heure_pleine_utc": "timestamp_utc"})
    )

    print(f"  → {len(df)} observations 15 min -> {len(agg)} heures")

    return agg


# COMBINER "DEF" ET "TR"


def combiner_def_et_tr(df_def, df_tr):

    print("\nCombinaison def + tr (def prioritaire, tr seulement pour combler les trous)...")

    heures_def = set(df_def["timestamp_utc"])
    chevauchement = df_tr["timestamp_utc"].isin(heures_def).sum()
    if chevauchement > 0:
        print(f"  → {chevauchement} heures présentes dans les deux fichiers : "
              f"on garde la version 'def'.")

    df_tr_complement = df_tr[~df_tr["timestamp_utc"].isin(heures_def)]
    print(f"  → {len(df_tr_complement)} heures ajoutées depuis 'tr' "
          f"(absentes de 'def').")

    df = pd.concat([df_def, df_tr_complement], ignore_index=True)
    df = df.sort_values("timestamp_utc").drop_duplicates(subset="timestamp_utc")

    return df



def appliquer_periode_cible(df):

    print(f"\nRestriction à la période cible {DATE_DEBUT} -> {DATE_FIN} (borne exclue)...")

    index_utc = pd.date_range(DATE_DEBUT, DATE_FIN, freq="1h", tz="UTC", inclusive="left")
    grille = pd.DataFrame({"timestamp_utc": index_utc})

    df = grille.merge(df, on="timestamp_utc", how="left")

    n_manquantes = df["consommation_mw"].isna().sum()
    print(f"  → grille de {len(df)} heures, {n_manquantes} heures sans donnée")

    return df


def gerer_valeurs_manquantes(df):

    print("\nGestion des valeurs manquantes résiduelles...")

    n_manquantes = df["consommation_mw"].isna().sum()
    print(f"  → {n_manquantes} valeurs manquantes à traiter")

    if n_manquantes > 0:
        df = df.sort_values("timestamp_utc")

        df["consommation_mw"] = df["consommation_mw"].interpolate(method="linear", limit=3)
        n_restantes = df["consommation_mw"].isna().sum()
        if n_restantes > 0:
            print(f"  ATTENTION : {n_restantes} valeurs restent manquantes "
                  f"(trou > 3h), à examiner manuellement.")
        df["source_donnee"] = df["source_donnee"].fillna("interpole")

    return df


def ajouter_variables(df):

    print("\nAjout des variables calendaires et des périodes de confinement...")

    df["timestamp_paris"] = df["timestamp_utc"].dt.tz_convert("Europe/Paris")

    df["heure"] = df["timestamp_paris"].dt.hour
    df["jour_semaine"] = df["timestamp_paris"].dt.dayofweek
   
    df["jour_annee"] = df["timestamp_paris"].dt.dayofyear
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

    # covid19 : booléen, True sur n'importe laquelle des 3 périodes.
    # confinement_numero : 1, 2, 3 ou <NA>, pour distinguer leur impact
    # individuellement (le confinement 1 a un effet net sur la conso,
    # les confinements 2 et 3 un effet plus faible, mêlé à la
    # saisonnalité -> utile de pouvoir les traiter différemment).
    df["covid19"] = False
    df["confinement_numero"] = pd.array([None] * len(df), dtype="Int64")

    for numero, (debut, fin) in enumerate(PERIODES_COVID, start=1):
        masque = date_paris.between(debut, fin + " 23:59:59")
        df.loc[masque, "covid19"] = True
        df.loc[masque, "confinement_numero"] = numero
        print(f"  → confinement {numero} ({debut} au {fin}) : {masque.sum()} heures")

    print(f"  → total : {df['covid19'].sum()} heures classées en période de confinement")

    return df



if __name__ == "__main__":

    df_def = lire_fichier_brut(FICHIER_DEF, "cons-def")
    df_def = horariser(df_def, "cons-def")

    df_tr = lire_fichier_brut(FICHIER_TR, "tr")
    df_tr = horariser(df_tr, "tr")

    df = combiner_def_et_tr(df_def, df_tr)
    df = appliquer_periode_cible(df)
    df = gerer_valeurs_manquantes(df)
    df = ajouter_variables(df)

    df = df[[
        "timestamp_utc", "timestamp_paris", "consommation_mw", "source_donnee",
        "heure", "jour_semaine", "jour_annee", "mois", "annee", "saison",
        "covid19", "confinement_numero",
    ]]

    print("\nTable finale :", df.shape)
    print(df.head())

    print("\n--- Vérification rapide ---")
    print("Doublons de timestamp_utc :", df["timestamp_utc"].duplicated().sum())
    print("Valeurs manquantes restantes :", df["consommation_mw"].isna().sum())
    print("Plage :", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())
    print("\nRépartition par source :")
    print(df["source_donnee"].value_counts())
    print("\nRépartition par confinement_numero :")
    print(df["confinement_numero"].value_counts(dropna=False))

    chemin_sortie = PROCESSED_DIR / "consommation_electricite_horaire.csv"
    df.to_csv(chemin_sortie, index=False, encoding="utf-8")
    print(f"\nSauvegardé : {chemin_sortie}")