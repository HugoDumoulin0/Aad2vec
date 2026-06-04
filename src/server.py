#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:07:07 2026

@author: hugodumoulin
"""

import pandas as pd
import matplotlib.pyplot as plt

from shiny import ui, render, reactive

from config import COMPLETE_CORPUS_LABEL

from loaders import (
    available_labels,
    corpus_choices,
    load_run_metadata,
    load_model,
    load_global_model,
    load_model_for_corpus,
    load_sentences,
    load_sentences_for_corpus,
    pca_cache_label
)

from pca_w2v import (
    PCA_CACHE,
    GLOBAL_PCA_CACHE,
    compute_dynamic_pca_and_clusters,
    compute_dynamic_global_pca,
    add_pca_contributions,
    filter_top_contributors,
    select_words_by_pos_then_topn,
)

from cooccurrences import (
    COOC_CACHE,
    characteristic_contexts_for_word_cooc,
    compute_cooc_pca,
    compute_global_cooc_pca,
    project_word_across_subcorpora_cooc,
    top_similar_words_by_cooc,
    top_cooccurring_words,
)

from semantic_shift import (
    project_all_words_across_subcorpora,
    compute_word_spread_table,
    nearest_words_for_word_in_model,
    nearest_words_to_cluster,
)

from context_extraction import (
    characteristic_context_words_for_cluster,
    extract_sentences_from_characteristic_context_words,
    extract_windows_from_characteristic_context_words,
)

from utils import (
    get_cluster_colors,
    rescale_series,
    dataframe_to_simple_html_table,
)


def server(input, output, session):
    cluster_console = reactive.Value("ACP clusters : en attente.")
    shift_console = reactive.Value("Évolution d'un mot : en attente.")
    cooc_console = reactive.Value("Cooccurrences : en attente.")

    def token_display_column(run_name):
        meta = load_run_metadata(run_name)
        token_representation = (
            meta.get("training_config", {})
            .get("token_representation", "word")
        )
        return "lemma" if token_representation == "lemma" else "word"

    def rename_token_column_for_display(df, run_name):
        display_col = token_display_column(run_name)
        if display_col != "word" and "word" in df.columns:
            return df.rename(columns={"word": display_col})
        return df

    def selected_pos(input_value):
        values = tuple(input_value) if input_value is not None else tuple()
        if "ALL" in values:
            return tuple()
        return values

    # ---- hyperparamètres du run
    @output
    @render.text
    def run_info():
        run_name = input.run_name()
    
        if not run_name or run_name == "Aucun run disponible":
            return "Aucun run disponible."
    
        meta = load_run_metadata(run_name)
        cfg = meta.get("training_config", {})
    
        if not cfg:
            return f"Run : {run_name}"
    
        lines = [f"Run : {run_name}"]
    
        for k, v in cfg.items():
            lines.append(f"{k} = {v}")
    
        return "\n".join(lines)
    
    
    # ---- calculs 
    @reactive.calc
    def subcorpus_pca_data():
        run_name = input.run_name()
        label = input.corpus_label()

        if not run_name or run_name == "Aucun run disponible":
            cluster_console.set("ACP clusters : aucun run disponible.")
            return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}

        if not label or label == "Aucune PCA disponible":
            cluster_console.set("ACP clusters : aucune PCA disponible.")
            return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}

        model = load_model_for_corpus(run_name, label)
        label_key = pca_cache_label(label)

        allowed_pos = selected_pos(input.pos_filter_sub())
        top_n = int(input.top_words_sub())
        n_clusters = int(input.n_clusters_sub())

        cluster_console.set(
            "ACP clusters : calcul/projection en cours "
            f"({label}, POS={allowed_pos or 'ALL'}, top_n={top_n}, clusters={n_clusters})."
        )

        df, meta = compute_dynamic_pca_and_clusters(
            run_name=run_name,
            model=model,
            label=label_key,
            allowed_pos=allowed_pos,
            top_n=top_n,
            n_clusters=n_clusters,
            cluster_on_pca=True
        )

        cluster_console.set(
            "ACP clusters : calcul terminé "
            f"({meta.get('n_words', len(df))} mots, "
            f"{meta.get('n_clusters', n_clusters)} clusters)."
        )

        return df, meta
    
    
    @reactive.calc
    def global_pca_data():
        run_name = input.run_name()

        if not run_name or run_name == "Aucun run disponible":
            shift_console.set("Évolution d'un mot : aucun run disponible.")
            return pd.DataFrame(columns=["word", "x", "y"]), {}, None

        global_model = load_global_model(run_name)

        allowed_pos = selected_pos(input.pos_filter_global())
        top_n = int(input.top_words_global())

        shift_console.set(
            "Évolution d'un mot : calcul de l'ACP globale "
            f"(POS={allowed_pos or 'ALL'}, top_n={top_n})."
        )

        df, meta, pca = compute_dynamic_global_pca(
            run_name=run_name,
            global_model=global_model,
            allowed_pos=allowed_pos,
            top_n=top_n
        )

        shift_console.set(
            "Évolution d'un mot : ACP globale prête "
            f"({meta.get('n_words', len(df))} mots)."
        )

        return df, meta, pca

    @output
    @render.text
    def cluster_console_text():
        return cluster_console.get()

    @output
    @render.text
    def shift_console_text():
        return shift_console.get()
    
    
    
    # ---- mise à jour labels lors du changement de run
    @reactive.Effect
    def _update_labels_for_run():
        run_name = input.run_name()
    
        choices = corpus_choices(run_name)
    
        if choices and choices != ["Aucune PCA disponible"]:
            current = input.corpus_label()
            selected = current if current in choices else COMPLETE_CORPUS_LABEL
    
            ui.update_select(
                "corpus_label",
                choices=choices,
                selected=selected,
                session=session,
            )
        else:
            ui.update_select(
                "corpus_label",
                choices=["Aucune PCA disponible"],
                selected="Aucune PCA disponible",
                session=session,
            )


    # ---- clusters disponibles pour le corpus sélectionné
    @reactive.Effect
    def _update_cluster_choices():
        df, _ = subcorpus_pca_data()

        if df.empty or "cluster" not in df.columns:
            ui.update_select(
                "cluster_id",
                choices=[""],
                selected="",
                session=session,
            )
            return

        valid = df["cluster"].dropna()
        if len(valid) == 0:
            ui.update_select(
                "cluster_id",
                choices=[""],
                selected="",
                session=session,
            )
            return

        cluster_ids = sorted(valid.astype(int).unique().tolist())
        choices = [str(c) for c in cluster_ids]

        current = input.cluster_id()
        selected = current if current in choices else choices[0]

        ui.update_select(
            "cluster_id",
            choices=choices,
            selected=selected,
            session=session,
        )

    # ---- info corpus
    @output
    @render.text
    def info():
        run_name = input.run_name()
        label = input.corpus_label()
        if not label or label == "Aucune PCA disponible":
            return "Aucune PCA disponible."

        df, meta = subcorpus_pca_data()
        model = load_model_for_corpus(run_name, label)
        sentences = load_sentences_for_corpus(run_name, label)


        lines = [f"Corpus : {label}"]
        lines.append(f"Nombre de mots PCA : {meta.get('n_words', 'n/a')}")
        lines.append(f"Clusters disponibles : {meta.get('n_clusters', 'n/a')}")

        evr = meta.get("explained_variance_ratio", [])
        evr_txt = ", ".join([f"{x:.3f}" for x in evr]) if evr else "n/a"
        lines.append(f"Variance expliquée : {evr_txt}")

        lines.append(f"Phrases sauvegardées : {len(sentences)}")
        lines.append(f"Modèle chargé : {'oui' if model is not None else 'non'}")

        return "\n".join(lines)

    # ---- plot ACP
    
    @output
    @render.plot
    def pca_plot():
    
        label = input.corpus_label()
        selected_cluster = input.cluster_id()
    
        df, meta = subcorpus_pca_data()
    
        fig, ax = plt.subplots(figsize=(10, 10))
    
        if df.empty:
            ax.set_title("Aucune donnée disponible")
            return fig
    
        df = add_pca_contributions(df, meta)
        df = filter_top_contributors(df, pct=int(input.top_contrib_pct()))
        
        use_scale = bool(input.scale_by_contrib())

        df = df.copy()
        
        if "contrib_total" not in df.columns:
            df["contrib_total"] = 1.0
        
        if use_scale:
            df["point_size"] = rescale_series(df["contrib_total"], min_size=8, max_size=40)
            df["label_size"] = rescale_series(df["contrib_total"], min_size=7, max_size=20)
        else:
            df["point_size"] = 12.0
            df["label_size"] = 7.0
        
        # variance expliquée
        evr = meta.get("explained_variance_ratio", [])
    
        if len(evr) >= 2:
            xlab = f"Axe 1 ({evr[0] * 100:.1f} %)"
            ylab = f"Axe 2 ({evr[1] * 100:.1f} %)"
        else:
            xlab = "Axe 1"
            ylab = "Axe 2"
    
        if "cluster" in df.columns:
    
            cluster_ids = sorted(
                df["cluster"].dropna().astype(int).unique().tolist()
            )
            
            colors = get_cluster_colors(len(cluster_ids))
            
            for i, cid in enumerate(cluster_ids):
            
                color = colors[i]
                sub = df[df["cluster"] == cid]
                
                is_selected = (selected_cluster != "" and int(selected_cluster) == cid)
                
                ax.scatter(
                    sub["x"],
                    sub["y"],
                    s=sub["point_size"] * (1.35 if selected_cluster != "" and int(selected_cluster) == cid else 1.0),
                    color=color,
                    alpha=1.0 if selected_cluster != "" and int(selected_cluster) == cid else 0.9,
                    edgecolor="black" if selected_cluster != "" and int(selected_cluster) == cid else None,
                    linewidth=0.5 if selected_cluster != "" and int(selected_cluster) == cid else 0.0,
                    label=f"Cluster {cid}",
                )
                # if selected_cluster != "" and int(selected_cluster) == cid:
    
                #     ax.scatter(
                #         sub["x"],
                #         sub["y"],
                #         s=24,
                #         color=color,
                #         alpha=1.0,
                #         edgecolor="black",
                #         linewidth=0.5,
                #         label=f"Cluster {cid}",
                #     )
    
                # else:
    
                #     ax.scatter(
                #         sub["x"],
                #         sub["y"],
                #         s=12,
                #         color=color,
                #         alpha=0.9,
                #         label=f"Cluster {cid}",
                #     )
    
            # annotations
    
            if selected_cluster != "":
    
                selected_cluster_int = int(selected_cluster)
    
                df_other = df[df["cluster"] != selected_cluster_int]
                df_sel = df[df["cluster"] == selected_cluster_int]
    
                for _, row in df_other.iterrows():
    
                    ax.annotate(
                        row["word"],
                        (row["x"], row["y"]),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=float(row["label_size"]) * 0.9,
                        alpha=0.7,
                    )
    
                for _, row in df_sel.iterrows():
    
                    ax.annotate(
                        row["word"],
                        (row["x"], row["y"]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=float(row["label_size"]) * 0.9,
                        fontweight="bold",
                    )
    
            else:
    
                for _, row in df.iterrows():
    
                    ax.annotate(
                        row["word"],
                        (row["x"], row["y"]),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=5,
                        alpha=0.8,
                    )
    
            ax.legend(
                title="Clusters",
                fontsize=8,
                title_fontsize=9,
                loc="best",
                frameon=True,
            )
    
        else:
    
            ax.scatter(df["x"], df["y"], s=10)
    
            for _, row in df.iterrows():
    
                ax.annotate(
                    row["word"],
                    (row["x"], row["y"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
    
        ax.set_title(f"ACP - {label}")
    
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    
        ax.tick_params(axis="both", labelsize=8)
    
        return fig

    # ---- table PCA brute
    @output
    @render.data_frame
    def pca_table():
        df, meta = subcorpus_pca_data()

        df = add_pca_contributions(df, meta)
        df = filter_top_contributors(df, pct=int(input.top_contrib_pct()))

        cols = [
            c for c in
            ["word", "x", "y", "cluster", "contrib_dim1", "contrib_dim2", "contrib_total"]
            if c in df.columns
        ]

        return render.DataGrid(df[cols])

    # ---- mots du cluster sélectionné
    @output
    @render.data_frame
    def cluster_words_table():
        df, _ = subcorpus_pca_data()
        cluster_id = input.cluster_id()

        if df.empty or cluster_id == "":
            return render.DataGrid(pd.DataFrame(columns=["word", "x", "y", "cluster"]))

        sub = df[df["cluster"] == int(cluster_id)].copy()
        sub = sub.sort_values("word").reset_index(drop=True)

        return render.DataGrid(sub)

    # ---- voisins latents du cluster
    @output
    @render.data_frame
    def cluster_neighbors_table():
        label = input.corpus_label()
        cluster_id = input.cluster_id()

        df, _ = subcorpus_pca_data()
        if df.empty or cluster_id == "":
            return render.DataGrid(pd.DataFrame(columns=["word", "similarity"]))

        run_name = input.run_name()
        model = load_model_for_corpus(run_name, label)
        sentences = load_sentences_for_corpus(run_name, label)
        if model is None:
            return render.DataGrid(pd.DataFrame(columns=["word", "similarity"]))

        cluster_words = (
            df[df["cluster"] == int(cluster_id)]["word"]
            .dropna()
            .astype(str)
            .tolist()
        )

        df_neighbors = nearest_words_to_cluster(cluster_words, model, topn=25)
        return render.DataGrid(df_neighbors)

    # ---- contextes locaux
    
    # @output
    # @render.data_frame
    # def cluster_windows_table():
    #     label = input.corpus_label()
    #     cluster_id = input.cluster_id()

    #     df, _ = subcorpus_pca_data()
    #     if df.empty or cluster_id == "":
    #         return render.DataGrid(pd.DataFrame(columns=[
    #             "sentence_id", "start", "end", "center_word", "window_text",
    #             "matched_cluster_words", "n_matched", "sim_cluster", "score"
    #         ]))

    #     run_name = input.run_name()
    #     model = load_model(run_name, label)
    #     sentences = load_sentences(run_name, label)

    #     if model is None or not sentences:
    #         return render.DataGrid(pd.DataFrame(columns=[
    #             "sentence_id", "start", "end", "center_word", "window_text",
    #             "matched_cluster_words", "n_matched", "sim_cluster", "score"
    #         ]))

    #     cluster_words = (
    #         df[df["cluster"] == int(cluster_id)]["word"]
    #         .dropna()
    #         .astype(str)
    #         .tolist()
    #     )

    #     df_windows = extract_cluster_windows(
    #         sentences=sentences,
    #         cluster_words=cluster_words,
    #         model=model,
    #         window_size=int(input.window_size()),
    #         topn=int(input.topn_windows()),
    #         min_unique_cluster_words=int(input.min_cluster_words()),
    #         deduplicate=bool(input.dedup_windows()),
    #         use_cluster_bonus=bool(input.use_cluster_bonus()),
    #     )
        
    #     df_context_words = characteristic_context_words_for_cluster(
    #         cluster_words=cluster_words,
    #         model=model,
    #         topn=25,
    #         min_count=5,
    #         exclude_cluster_words=True,
    #         allowed_pos=None,   # ou {"NOUN", "ADJ"} par ex.
    #         use_cosine=True,
    #     )
        
    #     df_sentences = extract_sentences_from_characteristic_context_words(
    #         sentences=sentences,
    #         context_words_df=df_context_words,
    #         cluster_words=cluster_words,
    #         topn=40,
    #         require_cluster_word=True,
    #         min_context_matches=1,
    #         deduplicate=True,
    #     )

    #     return render.DataGrid(
    #         df_windows
    #                            )
    
    @output
    @render.data_frame
    def cluster_context_words_table():
        label = input.corpus_label()
        cluster_id = input.cluster_id()
    
        df, _ = subcorpus_pca_data()
        if df.empty or cluster_id == "":
            return render.DataGrid(pd.DataFrame(columns=[
                "context_word", "score", "frequency"
            ]))
    
        run_name = input.run_name()
        
        model = load_model_for_corpus(run_name, label)
        sentences = load_sentences_for_corpus(run_name, label)
    
        if model is None:
            return render.DataGrid(pd.DataFrame(columns=[
                "context_word", "score", "frequency"
            ]))
    
        cluster_words = (
            df[df["cluster"] == int(cluster_id)]["word"]
            .dropna()
            .astype(str)
            .tolist()
        )
    
        df_context_words = characteristic_context_words_for_cluster(
            cluster_words=cluster_words,
            model=model,
            topn=25,
            min_count=5,
            exclude_cluster_words=True,
            allowed_pos=None,
            use_cosine=True,
        )
    
        return render.DataGrid(df_context_words)
    
    @output
    @render.data_frame
    def cluster_context_sentences_table():
        label = input.corpus_label()
        cluster_id = input.cluster_id()
    
        df, _ = subcorpus_pca_data()
        if df.empty or cluster_id == "":
            return render.DataGrid(pd.DataFrame(columns=[
                "sentence_id",
                "sentence_text",
                "matched_context_words",
                "n_context_matched",
                "matched_cluster_words",
                "n_cluster_matched",
                "score",
            ]))
    
        run_name = input.run_name()
        model = load_model_for_corpus(run_name, label)
        sentences = load_sentences_for_corpus(run_name, label)
    
        if model is None or not sentences:
            return render.DataGrid(pd.DataFrame(columns=[
                "sentence_id",
                "sentence_text",
                "matched_context_words",
                "n_context_matched",
                "matched_cluster_words",
                "n_cluster_matched",
                "score",
            ]))
    
        cluster_words = (
            df[df["cluster"] == int(cluster_id)]["word"]
            .dropna()
            .astype(str)
            .tolist()
        )
    
        df_context_words = characteristic_context_words_for_cluster(
            cluster_words=cluster_words,
            model=model,
            topn=25,
            min_count=5,
            exclude_cluster_words=True,
            allowed_pos=None,
            use_cosine=True,
        )
    
        df_sentences = extract_sentences_from_characteristic_context_words(
            sentences=sentences,
            context_words_df=df_context_words,
            cluster_words=cluster_words,
            topn=40,
            require_cluster_word=True,
            min_context_matches=1,
            deduplicate=True,
        )
    
        return render.DataGrid(df_sentences)
    
    @output
    @render.data_frame
    def cluster_context_windows_table():
        label = input.corpus_label()
        cluster_id = input.cluster_id()
    
        df, _ = subcorpus_pca_data()
        if df.empty or cluster_id == "":
            return render.DataGrid(pd.DataFrame(columns=[
                "sentence_id",
                "start",
                "end",
                "center_word",
                "window_text",
                "matched_context_words",
                "n_context_matched",
                "matched_cluster_words",
                "n_cluster_matched",
                "score",
            ]))
    
        run_name = input.run_name()
        model = load_model_for_corpus(run_name, label)
        sentences = load_sentences_for_corpus(run_name, label)
    
        if model is None or not sentences:
            return render.DataGrid(pd.DataFrame(columns=[
                "sentence_id",
                "start",
                "end",
                "center_word",
                "window_text",
                "matched_context_words",
                "n_context_matched",
                "matched_cluster_words",
                "n_cluster_matched",
                "score",
            ]))
    
        cluster_words = (
            df[df["cluster"] == int(cluster_id)]["word"]
            .dropna()
            .astype(str)
            .tolist()
        )
    
        df_context_words = characteristic_context_words_for_cluster(
            cluster_words=cluster_words,
            model=model,
            topn=25,
            min_count=5,
            exclude_cluster_words=True,
            allowed_pos=None,
            use_cosine=True,
        )
        
        df_windows= extract_windows_from_characteristic_context_words(
            sentences=sentences,
            context_words_df=df_context_words,
            cluster_words=cluster_words,
            window_size=int(input.window_size()),
            topn=int(input.topn_windows()),
            require_cluster_word=True,
            min_context_matches=1,
            deduplicate=bool(input.dedup_windows()),
            non_overlapping=True,
        )
        
        return render.DataGrid(df_windows)
    
    
    # ---- sélecteur de mots
    @output
    @render.ui
    def word_selector_ui():
        df_global, _, df_positions_all = global_shift_data()
    
        if df_global.empty:
            return ui.p("Aucune donnée d'évolution disponible.")
    
        # uniquement les mots qui existent vraiment dans les positions alignées
        available = set(df_positions_all["word"].dropna().astype(str).tolist())
        words = [w for w in df_global["word"].dropna().astype(str).tolist() if w in available]
    
        return ui.input_selectize(
            "tracked_word",
            "Choisir un mot",
            choices=words,
            selected=words[0] if words else None,
            multiple=False
        )
    
    # ---- graphique d'évolution
    @output
    @render.plot
    def word_shift_plot():
        df_global, meta, df_positions_all = global_shift_data()
    
        fig, ax = plt.subplots(figsize=(10, 10))
    
        if df_global.empty:
            ax.set_title("Aucune ACP globale disponible")
            return fig
    
        word = input.tracked_word() if hasattr(input, "tracked_word") else None
        if not word:
            ax.set_title("Choisir un mot")
            return fig
    
        if word not in set(df_global["word"].astype(str).tolist()):
            ax.set_title("Le mot sélectionné n'est pas disponible avec le filtre global courant")
            return fig
    
        if bool(input.show_global_background()) and not df_global.empty:
            ax.scatter(
                df_global["x"],
                df_global["y"],
                s=8,
                alpha=0.12,
                color="lightgray"
            )
    
            if bool(input.show_global_labels()):
                for _, row in df_global.iterrows():
                    ax.annotate(
                        row["word"],
                        (row["x"], row["y"]),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=4,
                        alpha=0.28,
                    )
    
        sub = df_positions_all[df_positions_all["word"] == word].copy()
    
        if sub.empty:
            ax.set_title(f"Aucune position disponible pour '{word}'")
            return fig
    
        sub = sub.sort_values("subcorpus").reset_index(drop=True)
    
        colors = get_cluster_colors(len(sub))
    
        ax.scatter(
            sub["x"],
            sub["y"],
            s=80,
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )
    
        ax.plot(
            sub["x"],
            sub["y"],
            color="black",
            alpha=0.5,
            linewidth=1.0
        )
    
        if bool(input.show_word_labels()):
            for _, row in sub.iterrows():
                ax.annotate(
                    row["subcorpus"],
                    (row["x"], row["y"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold"
                )
    
        global_word = df_global[df_global["word"] == word]
        if not global_word.empty:
            gx = global_word.iloc[0]["x"]
            gy = global_word.iloc[0]["y"]
    
            ax.scatter([gx], [gy], s=120, marker="X", color="red")
            ax.annotate(
                f"{word} (global)",
                (gx, gy),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold"
            )
    
        evr = meta.get("explained_variance_ratio", [])
        if len(evr) >= 2:
            ax.set_xlabel(f"Axe 1 ({evr[0] * 100:.1f} %)")
            ax.set_ylabel(f"Axe 2 ({evr[1] * 100:.1f} %)")
        else:
            ax.set_xlabel("Axe 1")
            ax.set_ylabel("Axe 2")
    
        ax.set_title(f"Évolution sémantique de '{word}' entre sous-corpus")
        ax.tick_params(axis="both", labelsize=8)
    
        return fig
    
    
    # ---- word-shift table
    @output
    @render.data_frame
    def word_shift_table():
        df_global, _, df_positions_all = global_shift_data()
    
        word = input.tracked_word() if hasattr(input, "tracked_word") else None
        if not word:
            return render.DataGrid(pd.DataFrame(columns=[
                "subcorpus", "word", "x", "y",
                "cosine_similarity_to_global",
                "cosine_distance_to_global"
            ]))
    
        if word not in set(df_global["word"].astype(str).tolist()):
            return render.DataGrid(pd.DataFrame(columns=[
                "subcorpus", "word", "x", "y",
                "cosine_similarity_to_global",
                "cosine_distance_to_global"
            ]))
    
        sub = (
            df_positions_all[df_positions_all["word"] == word]
            .sort_values("cosine_distance_to_global", ascending=False)
            .reset_index(drop=True)
        )
    
        return render.DataGrid(sub)
    
    # ---- global shift calculus
    @reactive.calc
    def global_shift_data():
        run_name = input.run_name()
        df_global, meta, pca = global_pca_data()
        global_model = load_global_model(run_name)

        if df_global.empty or global_model is None or pca is None:
            shift_console.set("Évolution d'un mot : projection impossible avec les données courantes.")
            return df_global, meta, pd.DataFrame(columns=[
                "subcorpus", "word", "x", "y",
                "cosine_similarity_to_global",
                "cosine_distance_to_global"
            ])

        labels = available_labels(run_name)
        allowed_words = df_global["word"].dropna().astype(str).tolist()

        shift_console.set(
            "Évolution d'un mot : projection du mot global dans les sous-corpus "
            f"({len(labels)} sous-corpus, {len(allowed_words)} mots candidats)."
        )

        df_positions_all = project_all_words_across_subcorpora(
            pca=pca,
            global_model=global_model,
            labels=labels,
            allowed_words=allowed_words,
            run_name=run_name
        )

        shift_console.set(
            "Évolution d'un mot : trajectoires prêtes "
            f"({len(df_positions_all)} positions projetées)."
        )

        return df_global, meta, df_positions_all
    
    @output
    @render.data_frame
    def word_spread_table():
        _, _, df_positions_all = global_shift_data()
    
        df_spread = compute_word_spread_table(df_positions_all)
    
        if df_spread.empty:
            return render.DataGrid(pd.DataFrame(columns=[
                "word",
                "n_subcorpora",
                "max_pairwise_dist_2d",
                "mean_pairwise_dist_2d",
                "farthest_pair",
            ]), selection_mode="row")
    
        top_n = int(input.top_spread_words())
        df_spread = df_spread.head(top_n).copy()
    
        # arrondis pour lisibilité
        df_spread["max_pairwise_dist_2d"] = df_spread["max_pairwise_dist_2d"].round(4)
        df_spread["mean_pairwise_dist_2d"] = df_spread["mean_pairwise_dist_2d"].round(4)
    
        return render.DataGrid(df_spread, selection_mode="row")

    @reactive.Effect
    def _update_tracked_word_from_spread_selection():
        selected = word_spread_table.data_view(selected=True)

        if selected is None or selected.empty or "word" not in selected.columns:
            return

        word = str(selected.iloc[0]["word"])

        df_global, _, df_positions_all = global_shift_data()
        if df_global.empty or df_positions_all.empty:
            return

        available = set(df_positions_all["word"].dropna().astype(str).tolist())
        words = [w for w in df_global["word"].dropna().astype(str).tolist() if w in available]

        if word not in words:
            return

        ui.update_selectize(
            "tracked_word",
            choices=words,
            selected=word,
            session=session,
        )
    
    # ---- neighbors for word
    @output
    @render.ui
    def neighbors_all_subcorpora_ui():
        run_name = input.run_name()
        word = input.tracked_word() if hasattr(input, "tracked_word") else None
    
        if not run_name or run_name == "Aucun run disponible":
            return ui.p("Aucun run disponible.")
    
        if not word:
            return ui.p("Choisir un mot.")
    
        labels = available_labels(run_name)
        topn = int(input.neighbors_topn_all())
        layout_mode = input.neighbors_layout()
    
        cards = []
    
        for label in labels:
            model = load_model(run_name, label)
            df = nearest_words_for_word_in_model(
                    word,
                    model,
                    topn=topn,
                    top_n_freq_filter=int(input.neighbors_top_freq_filter())
                )
    
            card = ui.card(
                ui.card_header(f"Sous-corpus : {label}"),
                dataframe_to_simple_html_table(df)
            )
    
            if layout_mode == "grid":
                cards.append(ui.column(4, card))
            else:
                cards.append(ui.column(12, card))
    
        if not cards:
            return ui.p("Aucun sous-corpus disponible.")
    
        return ui.row(*cards)

    @output
    @render.ui
    def predictive_context_words_all_subcorpora_ui():
        run_name = input.run_name()
        word = input.tracked_word() if hasattr(input, "tracked_word") else None

        if not run_name or run_name == "Aucun run disponible":
            return ui.p("Aucun run disponible.")

        if not word:
            return ui.p("Choisir un mot.")

        labels = available_labels(run_name)
        topn = int(input.predictive_context_topn_all())
        min_count = int(input.predictive_context_min_count_all())
        layout_mode = input.neighbors_layout()
        display_col = token_display_column(run_name)

        cards = []

        for label in labels:
            model = load_model(run_name, label)
            df = characteristic_context_words_for_cluster(
                cluster_words=[word],
                model=model,
                topn=topn,
                min_count=min_count,
                exclude_cluster_words=True,
                allowed_pos=None,
                use_cosine=True,
            )

            if display_col == "lemma" and "context_word" in df.columns:
                df = df.rename(columns={"context_word": "context_lemma"})

            card = ui.card(
                ui.card_header(f"Sous-corpus : {label}"),
                dataframe_to_simple_html_table(df)
            )

            if layout_mode == "grid":
                cards.append(ui.column(4, card))
            else:
                cards.append(ui.column(12, card))

        if not cards:
            return ui.p("Aucun sous-corpus disponible.")

        return ui.row(*cards)

    # ---- cooccurrences calc
    @reactive.calc
    def cooc_pca_data():
        run_name = input.run_name()
        label = input.corpus_label_cooc()
    
        if (
            not run_name
            or run_name == "Aucun run disponible"
            or not label
            or label == "Aucun corpus disponible"
        ):
            return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}
    
        if label == "Corpus complet":
            sentences = []
            for sub_label in available_labels(run_name):
                sentences.extend(load_sentences(run_name, sub_label))
        else:
            sentences = load_sentences(run_name, label)

        cfg = load_run_metadata(run_name).get("training_config", {})
        min_count = int(cfg.get("min_count", 1))
    
        key = (
            run_name,
            label,
            int(input.cooc_window()),
            int(input.cooc_top_words()),
            int(input.cooc_n_clusters()),
            selected_pos(input.cooc_pos_filter()),
            min_count,
        )
    
        if key in COOC_CACHE:
            cooc_console.set("Cooccurrences : projection récupérée depuis le cache.")
            return COOC_CACHE[key]
    
        matrix_key = (
            run_name,
            label,
            "full_ppmi",
            int(input.cooc_window()),
            min_count,
        )

        def set_cooc_status(message):
            cooc_console.set(message)

        cooc_console.set("Cooccurrences : démarrage du calcul.")

        with ui.Progress(min=0, max=1) as progress:
            progress.set(0.15, message="Cooccurrences", detail="Préparation de la matrice complète...")
            df, meta = compute_cooc_pca(
                sentences=sentences,
                allowed_pos=selected_pos(input.cooc_pos_filter()),
                top_n=int(input.cooc_top_words()),
                window_size=int(input.cooc_window()),
                n_clusters=int(input.cooc_n_clusters()),
                min_count=min_count,
                cluster_on_pca=True,
                matrix_cache_key=matrix_key,
                status_callback=set_cooc_status,
            )
            progress.set(0.9, message="Cooccurrences", detail="Finalisation de l'ACP...")

        cooc_console.set(
            "Cooccurrences : calcul terminé "
            f"({meta.get('n_words', 0)} mots affichés, "
            f"{meta.get('n_vocab_total', 0)} mots dans la matrice complète, "
            f"min_count={meta.get('min_count', min_count)})."
        )
    
        COOC_CACHE[key] = (df, meta)
        return df, meta

    @output
    @render.text
    def cooc_console_text():
        return cooc_console.get()

    
    @reactive.Effect
    def _update_cooc_labels_for_run():
        run_name = input.run_name()
        labels = available_labels(run_name)
        cfg = load_run_metadata(run_name).get("training_config", {})
        run_window = int(cfg.get("window", 5))

        ui.update_numeric(
            "cooc_window",
            value=run_window,
            session=session,
        )
        ui.update_numeric(
            "cooc_window_size",
            value=run_window,
            session=session,
        )
    
        if labels:
            choices = ["Corpus complet"] + labels
            current = input.corpus_label_cooc()
            selected = current if current in choices else "Corpus complet"
    
            ui.update_select(
                "corpus_label_cooc",
                choices=choices,
                selected=selected,
                session=session,
            )

            detail_current = input.cooc_detail_corpus_label()
            detail_selected = detail_current if detail_current in labels else labels[0]
            ui.update_select(
                "cooc_detail_corpus_label",
                choices=labels,
                selected=detail_selected,
                session=session,
            )
        else:
            ui.update_select(
                "corpus_label_cooc",
                choices=["Aucun corpus disponible"],
                selected="Aucun corpus disponible",
                session=session,
            )
            ui.update_select(
                "cooc_detail_corpus_label",
                choices=["Aucun corpus disponible"],
                selected="Aucun corpus disponible",
                session=session,
            )
    
    @reactive.Effect
    def _update_cooc_cluster_choices():
        df, _ = cooc_pca_data()
    
        if df.empty or "cluster" not in df.columns:
            ui.update_select(
                "cluster_id_cooc",
                choices=[""],
                selected="",
                session=session,
            )
            return
    
        cluster_ids = sorted(df["cluster"].dropna().astype(int).unique().tolist())
        choices = [str(c) for c in cluster_ids]
    
        current = input.cluster_id_cooc()
        selected = current if current in choices else choices[0]
    
        ui.update_select(
            "cluster_id_cooc",
            choices=choices,
            selected=selected,
            session=session,
        )
    
    # ---- cooccurrences plot
    @output
    @render.plot
    def cooc_plot():
        df, meta = cooc_pca_data()
        selected_cluster = input.cluster_id_cooc()
    
        fig, ax = plt.subplots(figsize=(10, 10))
    
        if df.empty:
            ax.set_title("Aucune donnée cooccurrence")
            return fig
    
        df = add_pca_contributions(df, meta)
        df = filter_top_contributors(df, pct=int(input.top_contrib_pct_cooc()))
        df = df.copy()
    
        if bool(input.scale_by_contrib_cooc()):
            df["point_size"] = rescale_series(df["contrib_total"], min_size=8, max_size=40)
            df["label_size"] = rescale_series(df["contrib_total"], min_size=7, max_size=20)
        else:
            df["point_size"] = 15.0
            df["label_size"] = 6.0
    
        evr = meta.get("explained_variance_ratio", [])
        if len(evr) >= 2:
            ax.set_xlabel(f"Axe 1 ({evr[0] * 100:.1f} %)")
            ax.set_ylabel(f"Axe 2 ({evr[1] * 100:.1f} %)")
        else:
            ax.set_xlabel("Axe 1")
            ax.set_ylabel("Axe 2")
    
        cluster_ids = sorted(df["cluster"].dropna().astype(int).unique().tolist())
        colors = get_cluster_colors(len(cluster_ids))
    
        for i, cid in enumerate(cluster_ids):
            sub = df[df["cluster"] == cid]
            is_selected = selected_cluster != "" and int(selected_cluster) == cid
    
            ax.scatter(
                sub["x"],
                sub["y"],
                s=sub["point_size"] * (1.35 if is_selected else 1.0),
                color=colors[i],
                alpha=1.0 if is_selected else 0.85,
                edgecolor="black" if is_selected else None,
                linewidth=0.5 if is_selected else 0.0,
                label=f"Cluster {cid}",
            )
    
        if selected_cluster != "":
            selected_cluster_int = int(selected_cluster)
    
            df_other = df[df["cluster"] != selected_cluster_int]
            df_sel = df[df["cluster"] == selected_cluster_int]
    
            for _, row in df_other.iterrows():
                ax.annotate(
                    row["word"],
                    (row["x"], row["y"]),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=float(row["label_size"]) * 0.9,
                    alpha=0.55,
                )
    
            for _, row in df_sel.iterrows():
                ax.annotate(
                    row["word"],
                    (row["x"], row["y"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=float(row["label_size"]) * 0.9,
                    fontweight="bold",
                )
        else:
            for _, row in df.iterrows():
                ax.annotate(
                    row["word"],
                    (row["x"], row["y"]),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=float(row["label_size"]),
                    alpha=0.8,
                )
    
        ax.legend(
            title="Clusters",
            fontsize=8,
            title_fontsize=9,
            loc="best",
            frameon=True,
        )
    
        ax.set_title("Projection ACP des cooccurrences")
        ax.tick_params(axis="both", labelsize=8)
    
        return fig

    
    # ---- cooccurrences table
    @output
    @render.data_frame
    def cooc_table():
        df, meta = cooc_pca_data()
    
        df = add_pca_contributions(df, meta)
        df = filter_top_contributors(df, pct=int(input.top_contrib_pct_cooc()))
    
        cols = [
            c for c in
            ["word", "x", "y", "cluster", "contrib_dim1", "contrib_dim2", "contrib_total"]
            if c in df.columns
        ]
    
        return render.DataGrid(df[cols])

    # ---- cooccurrences cluster words
    @output
    @render.data_frame
    def cooc_cluster_words_table():
        df, _ = cooc_pca_data()
        cluster_id = input.cluster_id_cooc()
    
        if df.empty or cluster_id == "":
            return render.DataGrid(pd.DataFrame(columns=["word", "x", "y", "cluster"]))
    
        sub = df[df["cluster"] == int(cluster_id)].copy()
        sub = sub.sort_values("word").reset_index(drop=True)
    
        return render.DataGrid(sub)

    
    # ---- cooccurrences trajectory 
    @output
    @render.ui
    def word_selector_ui_cooc():
        df_global, _, df_positions_all = global_shift_data_cooc()
    
        if df_global.empty:
            return ui.p("Aucune donnée cooccurrence disponible.")
    
        available = set(df_positions_all["word"].dropna().astype(str).tolist())
        words = [w for w in df_global["word"].dropna().astype(str).tolist() if w in available]
    
        return ui.input_selectize(
            "tracked_word_cooc",
            "Choisir un mot",
            choices=words,
            selected=words[0] if words else None,
            multiple=False
        )
        
    @output
    @render.plot
    def word_shift_plot_cooc():
        df_global, meta, df_positions_all = global_shift_data_cooc()
    
        fig, ax = plt.subplots(figsize=(10, 10))
    
        if df_global.empty:
            ax.set_title("Aucune ACP cooccurrence disponible")
            return fig
    
        word = input.tracked_word_cooc()
        if not word:
            ax.set_title("Choisir un mot")
            return fig
    
        sub = df_positions_all[df_positions_all["word"] == word].copy()
    
        if sub.empty:
            ax.set_title(f"Aucune position disponible pour '{word}'")
            return fig
    
        # fond global
        if bool(input.show_global_background_cooc()) and not df_global.empty:
            ax.scatter(
                df_global["x"],
                df_global["y"],
                s=8,
                alpha=0.12,
                color="lightgray"
            )

            if bool(input.show_global_labels_cooc()):
                for _, row in df_global.iterrows():
                    ax.annotate(
                        row["word"],
                        (row["x"], row["y"]),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=4,
                        alpha=0.28,
                    )
        
        if bool(input.show_word_labels_cooc()):
            for _, row in sub.iterrows():
                ax.annotate(
                    row["subcorpus"],
                    (row["x"], row["y"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold"
                )


    
        sub = sub.sort_values("subcorpus").reset_index(drop=True)
    
        colors = get_cluster_colors(len(sub))
    
        ax.scatter(
            sub["x"],
            sub["y"],
            s=80,
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )
    
        ax.plot(
            sub["x"],
            sub["y"],
            color="black",
            alpha=0.5,
            linewidth=1.0
        )
        
        global_word = df_global[df_global["word"] == word]
        if not global_word.empty:
            gx = global_word.iloc[0]["x"]
            gy = global_word.iloc[0]["y"]
    
            ax.scatter([gx], [gy], s=120, marker="X", color="red")
            ax.annotate(
                f"{word} (global)",
                (gx, gy),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold"
            )
    
        evr = meta.get("explained_variance_ratio", [])
        if len(evr) >= 2:
            ax.set_xlabel(f"Axe 1 ({evr[0]*100:.1f} %)")
            ax.set_ylabel(f"Axe 2 ({evr[1]*100:.1f} %)")
        else:
            ax.set_xlabel("Axe 1")
            ax.set_ylabel("Axe 2")
    
        ax.set_title(f"Évolution cooccurrence de '{word}' entre sous-corpus")
    
        return fig
    
    @output
    @render.data_frame
    def word_shift_table_cooc():
        df_global, _, df_positions_all = global_shift_data_cooc()
    
        word = input.tracked_word_cooc()
    
        if not word:
            return render.DataGrid(pd.DataFrame(columns=[
                "subcorpus", "word", "x", "y",
                "cosine_similarity_to_global",
                "cosine_distance_to_global"
            ]))
    
        sub = (
            df_positions_all[df_positions_all["word"] == word]
            .sort_values("cosine_distance_to_global", ascending=False)
            .reset_index(drop=True)
        )
    
        return render.DataGrid(sub)

    @output
    @render.ui
    def cooc_similar_words_by_subcorpus_ui():
        run_name = input.run_name()
        word = input.tracked_word_cooc()

        if not run_name or run_name == "Aucun run disponible":
            return ui.p("Aucun run disponible.")

        if not word:
            return ui.p("Choisir un mot.")

        label = input.cooc_detail_corpus_label()
        labels = available_labels(run_name)
        if not label or label == "Aucun corpus disponible" or label not in labels:
            return ui.p("Aucun sous-corpus sélectionné.")

        sentences = load_sentences(run_name, label)
        df = top_similar_words_by_cooc(
            sentences=sentences,
            target_word=word,
            window_size=int(input.cooc_window_size()),
            topn=int(input.cooc_top_neighbors_n()),
            min_frequency=int(input.cooc_min_token_frequency()),
        )
        df = rename_token_column_for_display(df, run_name)

        return ui.card(
            ui.card_header(f"Sous-corpus : {label}"),
            dataframe_to_simple_html_table(df),
        )

    @output
    @render.ui
    def cooc_top_words_by_subcorpus_ui():
        run_name = input.run_name()
        word = input.tracked_word_cooc()

        if not run_name or run_name == "Aucun run disponible":
            return ui.p("Aucun run disponible.")

        if not word:
            return ui.p("Choisir un mot.")

        label = input.cooc_detail_corpus_label()
        labels = available_labels(run_name)
        if not label or label == "Aucun corpus disponible" or label not in labels:
            return ui.p("Aucun sous-corpus sélectionné.")

        topn = int(input.cooc_top_neighbors_n())
        window_size = int(input.cooc_window_size())

        sentences = load_sentences(run_name, label)
        df = top_cooccurring_words(
            sentences=sentences,
            target_word=word,
            window_size=window_size,
            topn=topn,
            min_frequency=int(input.cooc_min_token_frequency()),
        )
        df = rename_token_column_for_display(df, run_name)

        return ui.card(
            ui.card_header(f"Sous-corpus : {label}"),
            dataframe_to_simple_html_table(df),
        )

    @output
    @render.ui
    def cooc_characteristic_contexts_ui():
        run_name = input.run_name()
        word = input.tracked_word_cooc()

        if not run_name or run_name == "Aucun run disponible":
            return ui.p("Aucun run disponible.")

        if not word:
            return ui.p("Choisir un mot.")

        label = input.cooc_detail_corpus_label()
        labels = available_labels(run_name)
        if not label or label == "Aucun corpus disponible" or label not in labels:
            return ui.p("Aucun sous-corpus sélectionné.")

        topn_words = int(input.cooc_top_neighbors_n())
        topn_contexts = int(input.cooc_top_contexts_n())
        window_size = int(input.cooc_window_size())

        sentences = load_sentences(run_name, label)
        df = characteristic_contexts_for_word_cooc(
            sentences=sentences,
            target_word=word,
            window_size=window_size,
            topn_words=topn_words,
            topn_contexts=topn_contexts,
            min_frequency=int(input.cooc_min_token_frequency()),
        )

        return ui.card(
            ui.card_header(f"Sous-corpus : {label}"),
            dataframe_to_simple_html_table(df),
        )
    
    @output
    @render.data_frame
    def word_spread_table_cooc():
        _, _, df_positions_all = global_shift_data_cooc()
    
        df_spread = compute_word_spread_table(df_positions_all)
    
        if df_spread.empty:
            return render.DataGrid(pd.DataFrame(), selection_mode="row")
    
        top_n = int(input.top_spread_words())
    
        df_spread = df_spread.head(top_n).copy()
    
        df_spread["max_pairwise_dist_2d"] = df_spread["max_pairwise_dist_2d"].round(4)
        df_spread["mean_pairwise_dist_2d"] = df_spread["mean_pairwise_dist_2d"].round(4)
    
        return render.DataGrid(df_spread, selection_mode="row")

    @reactive.Effect
    def _update_tracked_word_cooc_from_spread_selection():
        selected = word_spread_table_cooc.data_view(selected=True)

        if selected is None or selected.empty or "word" not in selected.columns:
            return

        word = str(selected.iloc[0]["word"])

        df_global, _, df_positions_all = global_shift_data_cooc()
        if df_global.empty or df_positions_all.empty:
            return

        available = set(df_positions_all["word"].dropna().astype(str).tolist())
        words = [w for w in df_global["word"].dropna().astype(str).tolist() if w in available]

        if word not in words:
            return

        ui.update_selectize(
            "tracked_word_cooc",
            choices=words,
            selected=word,
            session=session,
        )
    
    @reactive.calc
    def global_shift_data_cooc():
        run_name = input.run_name()
    
        global_model = load_global_model(run_name)
        allowed_pos = selected_pos(input.cooc_shift_pos_filter())
        top_n = int(input.top_words_global())
    
        allowed_words = select_words_by_pos_then_topn(global_model, allowed_pos, top_n) if global_model else None
    
        if not allowed_words:
            return pd.DataFrame(), {}, pd.DataFrame()
    
        df_cooc_global, meta, pca, global_vectors, svd = compute_global_cooc_pca(
            run_name=run_name,
            allowed_words=allowed_words,
            window_size=int(input.cooc_window_size())
        )
    
        if df_cooc_global.empty:
            return df_cooc_global, meta, pd.DataFrame()
    
        labels = available_labels(run_name)
    
        df_positions_all = project_word_across_subcorpora_cooc(
            run_name=run_name,
            labels=labels,
            allowed_words=allowed_words,
            global_vectors=global_vectors,
            pca=pca,
            svd=svd,
            window_size=int(input.cooc_window_size())
        )
    
        return df_cooc_global, meta, df_positions_all
    
    # ---- info cache
    @output
    @render.text
    def cache_info():
        return (
            f"ACP sous-corpus en cache : {len(PCA_CACHE)}\n"
            f"ACP globales en cache : {len(GLOBAL_PCA_CACHE)}"
    )
