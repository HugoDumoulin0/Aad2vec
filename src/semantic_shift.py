#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:54:29 2026

@author: hugodumoulin
"""

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from loaders import load_model
from utils import normalize_rows, cluster_centroid


#############################################
# Alignement Procrustes (Hamilton et al 2016)
#############################################
def align_submodel_to_global(
    global_model,
    sub_model,
    anchor_words=None,
    anchor_topn=5000,
    min_anchor=50
):
    """
    Aligne un modèle de sous-corpus sur le modèle global par orthogonal Procrustes.
    Si anchor_words est fourni, on aligne uniquement sur cet ensemble d'ancres.
    """

    global_vocab = set(global_model.wv.key_to_index.keys())
    sub_vocab = set(sub_model.wv.key_to_index.keys())

    common = list(global_vocab & sub_vocab)

    if anchor_words is not None:
        anchor_words = set(anchor_words)
        common = [w for w in common if w in anchor_words]
    else:
        common = sorted(
            common,
            key=lambda w: global_model.wv.get_vecattr(w, "count"),
            reverse=True
        )[:anchor_topn]

    if len(common) < min_anchor:
        raise ValueError(
            f"Pas assez de mots ancres pour l'alignement ({len(common)} < {min_anchor})"
        )

    X_sub = np.vstack([sub_model.wv[w] for w in common])
    X_glob = np.vstack([global_model.wv[w] for w in common])

    X_sub = normalize_rows(X_sub)
    X_glob = normalize_rows(X_glob)

    R, _ = orthogonal_procrustes(X_sub, X_glob)
    return R, common


def project_all_words_across_subcorpora(pca, global_model, labels, run_name, allowed_words=None):
    rows = []

    if global_model is None or pca is None:
        return pd.DataFrame(columns=[
            "subcorpus", "word", "x", "y",
            "cosine_similarity_to_global",
            "cosine_distance_to_global"
        ])

    allowed_words = set(allowed_words) if allowed_words is not None else None

    for label in labels:
        sub_model = load_model(run_name, label)
        if sub_model is None:
            continue

        try:
            R, common = align_submodel_to_global(
                global_model,
                sub_model,
                anchor_words=allowed_words,
                min_anchor=20
            )
        except Exception as e:
            print(f"[WARN] Alignement impossible pour {label}: {e}")
            continue

        if allowed_words is not None:
            common = [w for w in common if w in allowed_words]

        if not common:
            continue

        V_sub = np.vstack([sub_model.wv[w] for w in common])
        V_sub = normalize_rows(V_sub)
        V_sub_aligned = V_sub @ R

        XY = pca.transform(V_sub_aligned)

        V_global = np.vstack([global_model.wv[w] for w in common])
        V_global = normalize_rows(V_global)

        cos_sims = np.sum(V_sub_aligned * V_global, axis=1)
        cos_dists = 1.0 - cos_sims

        for i, w in enumerate(common):
            rows.append({
                "subcorpus": label,
                "word": w,
                "x": XY[i, 0],
                "y": XY[i, 1],
                "cosine_similarity_to_global": float(cos_sims[i]),
                "cosine_distance_to_global": float(cos_dists[i]),
            })

    return pd.DataFrame(rows)

#############################################
# Écarts
#############################################
def compute_word_spread_table(df_positions_all):
    """
    Calcule, pour chaque mot, sa dispersion entre sous-corpus
    sur le plan ACP 2D déjà projeté.
    """
    if df_positions_all.empty:
        return pd.DataFrame(columns=[
            "word",
            "n_subcorpora",
            "max_pairwise_dist_2d",
            "mean_pairwise_dist_2d",
            "farthest_pair",
        ])

    rows = []

    for word, grp in df_positions_all.groupby("word"):
        grp = grp.reset_index(drop=True)

        if len(grp) < 2:
            continue

        coords = grp[["x", "y"]].to_numpy(dtype=float)
        labels = grp["subcorpus"].astype(str).tolist()

        # matrice des distances euclidiennes 2D
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))

        # triangle supérieur sans diagonale
        iu = np.triu_indices(len(coords), k=1)
        pairwise = dists[iu]

        if len(pairwise) == 0:
            continue

        max_idx = int(np.argmax(pairwise))
        max_dist = float(pairwise[max_idx])
        mean_dist = float(np.mean(pairwise))

        i_max = iu[0][max_idx]
        j_max = iu[1][max_idx]

        rows.append({
            "word": word,
            "n_subcorpora": len(grp),
            "max_pairwise_dist_2d": max_dist,
            "mean_pairwise_dist_2d": mean_dist,
            "farthest_pair": f"{labels[i_max]} ↔ {labels[j_max]}",
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.sort_values(
        ["max_pairwise_dist_2d", "mean_pairwise_dist_2d"],
        ascending=[False, False]
    ).reset_index(drop=True)


def get_top_vocab_set(model, top_n_freq=None):
    if model is None:
        return set()

    words = list(model.wv.index_to_key)

    if top_n_freq is None or int(top_n_freq) <= 0:
        return set(words)

    return set(words[:int(top_n_freq)])


def nearest_words_for_word_in_model(word, model, topn=20, top_n_freq_filter=None):
    """
    Retourne les plus proches voisins d'un mot dans un modèle donné,
    éventuellement filtrés parmi les N mots les plus fréquents du corpus.
    Ajoute la fréquence brute et le rang fréquentiel du voisin.
    """
    if model is None or word is None or word not in model.wv:
        return pd.DataFrame(columns=["word", "similarity", "frequency", "freq_rank"])

    allowed = get_top_vocab_set(model, top_n_freq_filter)
    vocab_rank = {w: i + 1 for i, w in enumerate(model.wv.index_to_key)}

    raw = model.wv.most_similar(word, topn=max(topn * 5, 100))

    rows = []
    for neigh, sim in raw:
        if neigh not in allowed:
            continue

        freq = int(model.wv.get_vecattr(neigh, "count"))
        rank = int(vocab_rank.get(neigh, -1))

        rows.append((neigh, round(float(sim), 4), freq, rank))

        if len(rows) >= topn:
            break

    return pd.DataFrame(rows, columns=["word", "similarity", "frequency", "freq_rank"])


def nearest_words_to_cluster(cluster_words, model, topn=20):
    """
    Mots les plus proches du centroïde du cluster.
    """
    if model is None:
        return pd.DataFrame(columns=["word", "similarity"])

    cluster_words = [w for w in cluster_words if w in model.wv]
    if not cluster_words:
        return pd.DataFrame(columns=["word", "similarity"])

    centroid = cluster_centroid(cluster_words, model)
    if centroid is None:
        return pd.DataFrame(columns=["word", "similarity"])

    sims = model.wv.similar_by_vector(centroid, topn=topn + len(cluster_words) + 20)

    rows = []
    seen = set(cluster_words)

    for word, sim in sims:
        if word in seen:
            continue
        rows.append({"word": word, "similarity": round(float(sim), 4)})
        if len(rows) >= topn:
            break

    return pd.DataFrame(rows)

