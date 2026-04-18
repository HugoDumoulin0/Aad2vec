import argparse
import re
import unicodedata
import pandas as pd


def normalize_text(s):
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_mapping_dict(mapping_df):
    required_cols = {"source_label", "target_label"}
    missing = required_cols - set(mapping_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans la table de recodage: {sorted(missing)}")

    mapping_df = mapping_df.copy()
    mapping_df["source_norm"] = mapping_df["source_label"].apply(normalize_text)
    mapping_df["target_label"] = mapping_df["target_label"].astype(str).str.strip()

    # 1) on supprime les doublons exacts source_norm + target_label
    mapping_df = mapping_df.drop_duplicates(subset=["source_norm", "target_label"])

    # 2) on cherche les conflits : même source_norm -> plusieurs target_label différentes
    conflicts = (
        mapping_df.groupby("source_norm")["target_label"]
        .nunique()
        .reset_index(name="n_targets")
    )

    conflicts = conflicts[conflicts["n_targets"] > 1]

    if not conflicts.empty:
        conflict_rows = mapping_df[
            mapping_df["source_norm"].isin(conflicts["source_norm"])
        ].sort_values(["source_norm", "target_label"])

        raise ValueError(
            "Conflits détectés dans la table de recodage après normalisation.\n"
            "Une même modalité source normalisée pointe vers plusieurs cibles.\n\n"
            + conflict_rows.to_string(index=False)
        )

    # 3) on garde une seule ligne par source_norm
    mapping_df = mapping_df.drop_duplicates(subset=["source_norm"], keep="first")

    return dict(zip(mapping_df["source_norm"], mapping_df["target_label"]))


def main():
    parser = argparse.ArgumentParser(description="Ajoute une colonne recodée à un CSV source.")
    parser.add_argument("--input", required=True, help="CSV source")
    parser.add_argument("--mapping", required=True, help="Table de recodage CSV")
    parser.add_argument("--column", required=True, help="Nom de la colonne à recoder")
    parser.add_argument(
        "--new-column",
        default=None,
        help="Nom de la nouvelle colonne recodée. Par défaut: <colonne>_recode"
    )
    parser.add_argument(
        "--output",
        default="source_recode.csv",
        help="Nom du CSV enrichi en sortie"
    )
    parser.add_argument(
        "--unmapped",
        default="libelles_non_recodes.csv",
        help="CSV des modalités non recodées"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8", sep=";")
    mapping_df = pd.read_csv(args.mapping, encoding="utf-8")

    if args.column not in df.columns:
        raise ValueError(
            f"Colonne '{args.column}' absente du CSV source. Colonnes disponibles: {list(df.columns)}"
        )

    new_col = args.new_column if args.new_column else f"{args.column}_recode"

    mapping = build_mapping_dict(mapping_df)

    # normalisation de la colonne source
    norm_col = f"{args.column}_norm_temp_internal"
    df[norm_col] = df[args.column].apply(normalize_text)

    # ajout de la nouvelle colonne recodée
    df[new_col] = df[norm_col].map(mapping)

    # option prudente : si pas de recodage trouvé, on laisse NA
    # si vous préférez conserver la valeur d'origine quand pas trouvé, remplacez par :
    # df[new_col] = df[new_col].fillna(df[args.column])

    # export des non recodés
    unmapped = (
        df[df[new_col].isna()]
        .groupby(args.column, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", args.column], ascending=[False, True])
    )
    unmapped.to_csv(args.unmapped, index=False, encoding="utf-8-sig")

    # suppression de la colonne technique
    df = df.drop(columns=[norm_col])

    # export final
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    total = len(df)
    recoded = df[new_col].notna().sum()
    not_recoded = df[new_col].isna().sum()

    print("Terminé.")
    print(f"CSV source          : {args.input}")
    print(f"Colonne source      : {args.column}")
    print(f"Nouvelle colonne    : {new_col}")
    print(f"Lignes totales      : {total}")
    print(f"Lignes recodées     : {recoded}")
    print(f"Lignes non recodées : {not_recoded}")
    print(f"Sortie enrichie     : {args.output}")
    print(f"Non recodés         : {args.unmapped}")


if __name__ == "__main__":
    main()