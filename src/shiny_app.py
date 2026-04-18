#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:45:34 2026

@author: hugodumoulin
"""
from shiny import App, ui, reactive, render
import os
import json
import pandas as pd

CORPUS_PATH = "././corpus"
METADATA_PATH = os.path.join(CORPUS_PATH, "metadata.csv")
EXPORT_JSON = "subcorpora.json"


def normalize_filename(x):
    if pd.isna(x):
        return None

    x = str(x).strip()
    if not x:
        return None

    # enlève le chemin éventuel
    x = os.path.basename(x)

    # enlève l'extension éventuelle
    x = os.path.splitext(x)[0]

    # normalisation légère
    x = x.lower().strip()

    return x


def list_corpus_files():
    if not os.path.isdir(CORPUS_PATH):
        return []

    return sorted(
        [
            f for f in os.listdir(CORPUS_PATH)
            if os.path.isfile(os.path.join(CORPUS_PATH, f))
            and f.lower() != "metadata.csv"
        ]
    )


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    return pd.read_csv(METADATA_PATH, sep=";")


def detect_file_column(df, corpus_files):
    if df is None or df.empty:
        return None

    candidates = [
        "filename", "file", "fichier", "nom_fichier",
        "document", "doc", "source", "texte", "id"
    ]

    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]

    corpus_norm = {normalize_filename(f) for f in corpus_files}

    best_col = None
    best_score = -1

    for col in df.columns:
        values = {
            normalize_filename(v)
            for v in df[col].dropna().astype(str)
        }
        values.discard(None)

        score = len(values & corpus_norm)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col if best_score > 0 else None


def build_groups_from_metadata(df, file_col, segment_col, corpus_files):
    groups = {}

    if df is None or file_col is None or segment_col is None:
        return groups

    # mapping nom normalisé -> vrai nom de fichier
    corpus_lookup = {
        normalize_filename(f): f
        for f in corpus_files
    }

    matched = 0
    unmatched_examples = []

    for _, row in df.iterrows():
        raw_file = row.get(file_col)
        raw_group = row.get(segment_col)

        if pd.isna(raw_file) or pd.isna(raw_group):
            continue

        file_norm = normalize_filename(raw_file)
        group_name = str(raw_group).strip()

        if not file_norm or not group_name:
            continue

        if file_norm not in corpus_lookup:
            if len(unmatched_examples) < 10:
                unmatched_examples.append(str(raw_file))
            continue

        real_filename = corpus_lookup[file_norm]
        groups.setdefault(group_name, []).append(real_filename)
        matched += 1

    for k in groups:
        groups[k] = sorted(set(groups[k]))

    print(f"[DEBUG] lignes appariées : {matched}")
    if unmatched_examples:
        print("[DEBUG] exemples non appariés :", unmatched_examples)

    return dict(sorted(groups.items(), key=lambda x: x[0].lower()))


def groups_to_dataframe(groups):
    rows = []

    for group, files in sorted(groups.items(), key=lambda x: x[0].lower()):
        rows.append(
            {
                "groupe": group,
                "nb_fichiers": len(files),
                "fichiers": ", ".join(files)
            }
        )

    if not rows:
        return pd.DataFrame(columns=["groupe", "nb_fichiers", "fichiers"])

    return pd.DataFrame(rows)


app_ui = ui.page_fluid(
    ui.h1("Aad2Vec"),
    ui.h2("Configuration"),
    
    ui.hr(),

    ui.h3("Paramètres d'entraînement Word2Vec"),

    ui.row(
        ui.column(
            2,
            ui.input_numeric("vector_size", "Dim", value=300, min=50, max=1000)
        ),
        ui.column(
            2,
            ui.input_numeric("window", "Window", value=5, min=1, max=20)
        ),
        ui.column(
            2,
            ui.input_numeric("min_count", "Min count", value=5, min=1, max=50)
        ),
        ui.column(
            2,
            ui.input_numeric("epochs", "Epochs", value=50, min=1, max=500)
        ),
        ui.column(
            2,
            ui.input_select(
                "sg",
                "Architecture",
                choices={"1": "Skip-gram", "0": "CBOW"},
                selected="1"
            )
        ),
    ),

    ui.row(
        ui.column(
            2,
            ui.input_numeric("negative", "Negative", value=5, min=1, max=50)
        ),
        ui.column(
            2,
            ui.input_numeric("sample", "Sample", value=0.001, min=0.0, max=0.1, step=0.0001)
        ),
        ui.column(
            2,
            ui.input_checkbox("lowercase", "Tout en minuscules", value=True)
        ),
    ),

    ui.hr(),
    
    ui.h3("Paramètres de partition du corpus"),

    ui.row(
        ui.column(
            4,
            ui.input_select(
                "mode",
                "Mode de partition",
                choices={
                    "automatique": "Automatique (metadata.csv)",
                    "manuel": "Manuel",
                },
                selected="automatique",
            ),
        )
    ),

    ui.output_ui("mode_controls"),

    ui.hr(),

    ui.row(
        ui.column(3, ui.input_action_button("save", "Exporter JSON")),
        ui.column(3, ui.input_action_button("reset_groups", "Réinitialiser les groupes")),
    ),

    ui.hr(),

    ui.h4("Statut"),
    ui.output_text("status_text"),

    ui.h4("Résumé des parties"),
    ui.output_data_frame("groups_table"),

    ui.h4("Aperçu JSON"),
    ui.output_text_verbatim("json_preview"),
)


def server(input, output, session):
    groups = reactive.Value({})
    status_msg = reactive.Value("En attente...")

    @output
    @render.ui
    def mode_controls():
        mode = input.mode()
        corpus_files = list_corpus_files()

        if mode == "manuel":
            return ui.TagList(
                ui.h4("Mode manuel"),
                ui.input_checkbox_group(
                    "files",
                    "Choisir les fichiers",
                    choices=corpus_files
                ),
                ui.input_text("label", "Nom du groupe"),
                ui.input_action_button("add_manual", "Ajouter au groupe")
            )

        df = load_metadata()

        if df is None:
            return ui.TagList(
                ui.h4("Mode automatique"),
                ui.p("Aucun fichier metadata.csv trouvé dans le dossier ./corpus.")
            )

        file_col = detect_file_column(df, corpus_files)

        if file_col is None:
            return ui.TagList(
                ui.h4("Mode automatique"),
                ui.p("Impossible d'identifier la colonne des noms de fichiers dans metadata.csv."),
                ui.p(
                    "Noms attendus de préférence : filename, file, fichier, "
                    "nom_fichier, document, source."
                )
            )

        segment_choices = [c for c in df.columns if c != file_col]

        if not segment_choices:
            return ui.TagList(
                ui.h4("Mode automatique"),
                ui.p("Aucune colonne de partition disponible dans metadata.csv.")
            )

        return ui.TagList(
            ui.h4("Mode automatique"),
            ui.p(f"Colonne fichier détectée : {file_col}"),
            ui.input_select(
                "segment_col",
                "Métadonnée de partition",
                choices=segment_choices,
                selected=segment_choices[0]
            ),
            ui.input_action_button(
                "build_auto",
                "Construire automatiquement les groupes"
            )
        )

    @output
    @render.text
    def status_text():
        return status_msg.get()

    @output
    @render.text
    def json_preview():
        return json.dumps(groups.get(), indent=2, ensure_ascii=False)

    @output
    @render.data_frame
    def groups_table():
        df = groups_to_dataframe(groups.get())
        return render.DataGrid(df)

    @reactive.Effect
    @reactive.event(input.add_manual)
    def _add_manual():
        if input.mode() != "manuel":
            return

        selected = input.files() if input.files() is not None else []
        label = input.label().strip() if input.label() is not None else ""

        if not label:
            status_msg.set("⚠️ Nom de groupe manquant.")
            return

        if not selected:
            status_msg.set("⚠️ Aucun fichier sélectionné.")
            return

        current = {k: list(v) for k, v in groups.get().items()}
        current.setdefault(label, [])
        current[label].extend(selected)
        current[label] = sorted(set(current[label]))

        groups.set(current)
        status_msg.set(f"✅ Groupe '{label}' mis à jour avec {len(selected)} fichier(s).")

    @reactive.Effect
    @reactive.event(input.build_auto)
    def _build_auto():
        if input.mode() != "automatique":
            return

        corpus_files = list_corpus_files()
        df = load_metadata()

        if df is None:
            status_msg.set("⚠️ metadata.csv introuvable dans ./corpus.")
            return

        file_col = detect_file_column(df, corpus_files)
        segment_col = input.segment_col()

        print("[DEBUG] fichiers corpus :", corpus_files[:10])
        print("[DEBUG] colonnes metadata :", list(df.columns))
        print("[DEBUG] colonne fichier détectée :", file_col)
        print("[DEBUG] colonne de partition :", segment_col)

        if file_col is None:
            status_msg.set("⚠️ Impossible d'identifier la colonne des noms de fichiers.")
            return

        auto_groups = build_groups_from_metadata(df, file_col, segment_col, corpus_files)

        if not auto_groups:
            status_msg.set("⚠️ Aucun groupe généré. Vérifie les noms de fichiers et la colonne choisie.")
            return

        groups.set(auto_groups)
        status_msg.set(
            f"✅ partition automatique terminée selon '{segment_col}' "
            f"({len(auto_groups)} groupe(s))."
        )

    @reactive.Effect
    @reactive.event(input.reset_groups)
    def _reset_groups():
        groups.set({})
        status_msg.set("🧹 Groupes réinitialisés.")

    @reactive.Effect
    @reactive.event(input.save)
    def _save_json():
        subcorpora = groups.get()

        if not subcorpora:
            status_msg.set("⚠️ Aucun groupe à exporter.")
            return

        payload = {
            "subcorpora": subcorpora,
            "training_config": {
                "vector_size": int(input.vector_size()),
                "window": int(input.window()),
                "min_count": int(input.min_count()),
                "epochs": int(input.epochs()),
                "sg": int(input.sg()),
                "negative": int(input.negative()),
                "sample": float(input.sample()),
                "lowercase": bool(input.lowercase()),
            }
        }

        with open(EXPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        status_msg.set(f"💾 {EXPORT_JSON} sauvegardé avec succès.")


app = App(app_ui, server)