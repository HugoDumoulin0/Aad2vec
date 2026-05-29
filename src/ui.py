#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:04:26 2026

@author: hugodumoulin
"""

import os

from shiny import ui

from config import POS_OPTIONS, COMPLETE_CORPUS_LABEL
from loaders import available_runs, corpus_choices


runs0 = available_runs()
env_default_run = os.environ.get("DEFAULT_RUN_NAME")
default_run = env_default_run if env_default_run in runs0 else (runs0[0] if runs0 else None)
labels0 = corpus_choices(default_run) if default_run else []
detail_labels0 = [
    label for label in labels0
    if label not in [COMPLETE_CORPUS_LABEL, "Aucune PCA disponible"]
]

app_ui = ui.page_fluid(
    ui.h2("Visualisation sémantique"),
    
    ui.output_text_verbatim("cache_info"),
    
    ui.row(
    ui.column(
        8,
        ui.input_select(
            "run_name",
            "Run",
            choices=available_runs() if available_runs() else ["Aucun run disponible"],
            selected=default_run if default_run else "Aucun run disponible",
            ),
        ),
    ),
    ui.output_text_verbatim("run_info"),

    ui.navset_tab(
        ui.nav_panel(
            "ACP clusters",
            ui.row(
                ui.column(
                    4,
                    ui.input_select(
                        "corpus_label",
                        "Choisir une ACP",
                        choices=labels0 if labels0 else ["Aucune PCA disponible"],
                        selected=labels0[0] if labels0 else "Aucune PCA disponible",
                    ),
                ),
                ui.column(
                    2,
                    ui.input_action_button("refresh_labels", "Rafraîchir"),
                ),
            ),
        
            ui.hr(),
        
            ui.h4("Informations"),
            ui.output_text_verbatim("info"),
            
            ui.input_selectize(
                "pos_filter_sub",
                "POS affichés",
                choices=POS_OPTIONS,
                selected=["NOUN"],
                multiple=True
            ),
            ui.input_numeric(
                "top_words_sub",
                "Nombre de mots après filtrage POS",
                value=100,
                min=10,
                max=5000
            ),
            ui.input_numeric(
                "n_clusters_sub",
                "Nombre de clusters",
                value=8,
                min=2,
                max=30
            ),
        
            ui.h3("Aad2vec"),
            ui.h4("Graphique ACP"),
            ui.output_plot("pca_plot", height="800px"),
        
            ui.h4("Coordonnées"),
            ui.output_data_frame("pca_table"),
            
            ui.h4("Affichage ACP"),
            ui.input_slider(
            "top_contrib_pct",
            "Pourcentage de points affichés (par contribution)",
            min=5,
            max=100,
            value=100,
            step=5
            ),
            
            ui.input_checkbox(
            "scale_by_contrib",
            "Taille proportionnelle à la contribution",
            value=False
            ),
        
            ui.hr(),
        
            ui.h3("Exploration par cluster"),
        
            ui.row(
                ui.column(
                    3,
                    ui.input_select(
                        "cluster_id",
                        "Choisir un cluster",
                        choices=[""],
                        selected="",
                    ),
                ),
                ui.column(
                    2,
                    ui.input_numeric("window_size", "Fenêtre ± tokens", value=10, min=1, max=20),
                ),
                ui.column(
                    2,
                    ui.input_numeric("topn_windows", "Top contextes", value=40, min=5, max=200),
                ),
                ui.column(
                    2,
                    ui.input_numeric("min_cluster_words", "Min mots cluster/fenêtre", value=1, min=1, max=10),
                ),
                ui.column(
                    2,
                    ui.input_checkbox("dedup_windows", "Dédupliquer", value=True),
                ),
                ui.column(
                    2,
                    ui.input_checkbox("use_cluster_bonus", "Bonus mots cluster", value=True),
                    ),
                ui.column(
                    2,
                    ui.input_numeric("cluster_bonus_weight", "Poids bonus", value=0.20, min=0, max=5, step=0.05),
                ),
            ),
        
            ui.h4("Mots du cluster"),
            ui.output_data_frame("cluster_words_table"),
        
            ui.h4("Voisins latents du cluster"),
            ui.output_data_frame("cluster_neighbors_table"),
        
            # ui.h4("Contextes locaux explicatifs"),
            # ui.output_data_frame("cluster_windows_table"),
            
            ui.h4("Mots de contexte caractéristiques du cluster"),
            ui.output_data_frame("cluster_context_words_table"),

            # ui.h4("Phrases contenant ces contextes caractéristiques"),
            # ui.output_data_frame("cluster_context_sentences_table"),
            
            ui.h4("Fenêtres locales autour des contextes caractéristiques"),
            ui.output_data_frame("cluster_context_windows_table"),
            ),
        
        
        ui.nav_panel(
            "Évolution d’un mot",

            ui.h4("Projection globale alignée"),
            
            ui.input_selectize(
                "pos_filter_global",
                "POS affichés dans l'ACP globale",
                choices=POS_OPTIONS,
                selected=["NOUN"],
                multiple=True
            ),
            ui.input_numeric(
                "top_words_global",
                "Nombre de mots après filtrage POS (global)",
                value=200,
                min=10,
                max=20000
                ),
            
            ui.row(
                ui.column(
                    4,
                    ui.output_ui("word_selector_ui")
                ),
                ui.column(
                    3,
                    ui.input_checkbox(
                        "show_global_background",
                        "Afficher le fond global",
                        value=True
                    )
                ),
                ui.column(
                    3,
                    ui.input_checkbox(
                        "show_word_labels",
                        "Afficher les labels sous-corpus",
                        value=True
                    )
                ),
                ui.column(
                    3,
                    ui.input_checkbox(
                        "show_global_labels",
                        "Afficher les labels du fond global",
                        value=False
                    )
                ),
            ),  

            ui.h4("Déplacement sémantique du mot"),
            ui.output_plot("word_shift_plot", height="850px"),

            ui.h4("Distances au global"),
            ui.output_data_frame("word_shift_table"),
            ui.hr(),
            
            ui.hr(),
            ui.h4("Plus proches voisins du mot dans tous les sous-corpus"),
            
            ui.row(
                ui.column(
                    3,
                    ui.input_numeric(
                        "neighbors_topn_all",
                        "Top voisins par sous-corpus",
                        value=15,
                        min=5,
                        max=100
                    ),
                ),
                ui.column(
                    3,
                    ui.input_numeric(
                        "neighbors_top_freq_filter",
                        "Garder seulement parmi les N mots les plus fréquents",
                        value=500,
                        min=0,
                        max=500000
                    ),
                ),
                ui.column(
                    3,
                    ui.input_select(
                        "neighbors_layout",
                        "Disposition",
                        choices={
                            "grid": "Côte à côte",
                            "stack": "L'un en dessous de l'autre"
                        },
                        selected="grid"
                    ),
                ),
            ),
            ui.output_ui("neighbors_all_subcorpora_ui"),
            
            ui.h4("Mots les plus dispersés entre sous-corpus"),
            ui.input_numeric(
                "top_spread_words",
                "Nombre de mots à afficher",
                value=30,
                min=5,
                max=500
            ),
            ui.output_data_frame("word_spread_table"),
            ),
        
    ui.nav_panel("Cooccurrences",

    ui.h4("Projection par cooccurrences"),
    
    ui.row(
            ui.column(
                4,
                ui.input_select(
                    "corpus_label_cooc",
                    "Choisir un corpus",
                    choices=labels0 if labels0 else ["Aucun corpus disponible"],
                    selected=COMPLETE_CORPUS_LABEL if labels0 else "Aucun corpus disponible",

                ),
            ),
        ),

    ui.row(
        ui.column(
            3,
            ui.input_numeric(
                "cooc_window",
                "Fenêtre ± tokens",
                value=5,
                min=1,
                max=20
            ),
        ),



        ui.column(
            3,
            ui.input_numeric(
                "cooc_top_words",
                "Top mots",
                value=300,
                min=50,
                max=5000
            ),
        ),
        ui.column(
            4,
            ui.input_selectize(
                "cooc_pos_filter",
                "POS affichés",
                choices=POS_OPTIONS,
                selected=["NOUN"],
                multiple=True
            ),
        ),
    ),

    ui.hr(),
    
    ui.column(
    3,
    ui.input_numeric(
        "cooc_n_clusters",
        "Nombre de clusters",
        value=8,
        min=2,
        max=30
    ),
),
    
    

    ui.h4("ACP cooccurrences"),
    ui.output_plot("cooc_plot", height="800px"),
    
    ui.h4("Affichage ACP cooccurrences"),
    
    ui.input_slider(
        "top_contrib_pct_cooc",
        "Pourcentage de points affichés (par contribution)",
        min=5,
        max=100,
        value=100,
        step=5
    ),
    
    ui.input_checkbox(
        "scale_by_contrib_cooc",
        "Taille proportionnelle à la contribution",
        value=False
    ),
    
    ui.input_select(
        "cluster_id_cooc",
        "Choisir un cluster",
        choices=[""],
        selected="",
    ),

    ui.h4("Coordonnées"),
    ui.output_data_frame("cooc_table"),
    
    ui.h4("Mots du cluster cooccurrences"),
    ui.output_data_frame("cooc_cluster_words_table"),

),
        
        
        
        ui.nav_panel(
    "Évolution (cooccurrences)",
    
    ui.row(
    ui.column(
        3,
        ui.input_checkbox(
            "show_global_background_cooc",
            "Afficher le fond global",
            value=True
        )
    ),
    ui.column(
        3,
        ui.input_checkbox(
            "show_word_labels_cooc",
            "Afficher les labels sous-corpus",
            value=True
        )
    ),
    ui.column(
        3,
        ui.input_checkbox(
            "show_global_labels_cooc",
            "Afficher les labels du fond global",
            value=False
        )
    ),
),


    ui.input_numeric(
        "cooc_window_size",
        "Fenêtre cooccurrences",
        value=4,
        min=1,
        max=10
    ),

    ui.output_ui("word_selector_ui_cooc"),

    ui.h4("Trajectoire cooccurrences"),
    ui.output_plot("word_shift_plot_cooc", height="850px"),

    ui.h4("Distances au global"),
    ui.output_data_frame("word_shift_table_cooc"),

    ui.input_select(
        "cooc_detail_corpus_label",
        "Sous-corpus affiché",
        choices=detail_labels0 if detail_labels0 else ["Aucun corpus disponible"],
        selected=detail_labels0[0] if detail_labels0 else "Aucun corpus disponible",
    ),
    ui.input_numeric(
        "cooc_top_neighbors_n",
        "Nombre de mots",
        value=15,
        min=5,
        max=100
    ),

    ui.h4("Mots les plus similaires par sous-corpus (matrice PPMI)"),
    ui.output_ui("cooc_similar_words_by_subcorpus_ui"),

    ui.h4("Mots les plus cooccurrents par sous-corpus (rang PPMI)"),
    ui.output_ui("cooc_top_words_by_subcorpus_ui"),

    ui.h4("Contextes caractéristiques"),
    ui.input_numeric(
        "cooc_top_contexts_n",
        "Contextes par sous-corpus",
        value=10,
        min=3,
        max=50
    ),
    ui.output_ui("cooc_characteristic_contexts_ui"),

    ui.h4("Mots les plus dispersés"),
    ui.output_data_frame("word_spread_table_cooc"),
)
    )
    )
