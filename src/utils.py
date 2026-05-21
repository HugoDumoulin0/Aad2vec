#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:32:36 2026

@author: hugodumoulin
"""

import hashlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shiny import ui




def stable_short_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

def sentence_tokens(sent):
    if isinstance(sent, dict):
        return sent.get("tokens", [])
    return sent

def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def rescale_series(values, min_size, max_size):
    """
    Ramène une série numérique dans un intervalle [min_size, max_size].
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if vmax == vmin:
        return np.full(len(values), (min_size + max_size) / 2.0)
    
    scaled = (values - vmin) / (vmax - vmin)
    
    #exagération
    exponent = 1.8
    scaled = scaled ** exponent
    return min_size + scaled * (max_size - min_size)

# ============================================================
# vecteurs
# ============================================================

def cosine(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def mean_vector(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def cluster_centroid(cluster_words, model):
    vecs = [model.wv[w] for w in cluster_words if w in model.wv]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

# ============================================================
# utilitaires visualisation
# ============================================================

def dataframe_to_simple_html_table(df):
    """
    Convertit un DataFrame en petit tableau HTML simple pour affichage dans une card.
    """
    if df.empty:
        return ui.p("Aucune donnée disponible.")

    header = ui.tags.tr(
        *[ui.tags.th(col, style="padding: 4px 8px; text-align: left;") for col in df.columns]
    )

    rows = []
    for _, row in df.iterrows():
        rows.append(
            ui.tags.tr(
                *[
                    ui.tags.td(str(row[col]), style="padding: 4px 8px; vertical-align: top;")
                    for col in df.columns
                ]
            )
        )

    return ui.tags.table(
        ui.tags.thead(header),
        ui.tags.tbody(*rows),
        style="width: 100%; border-collapse: collapse; font-size: 0.95em;"
    )


def get_cluster_colors(n):
    """
    Génère n couleurs
    """
    if n <= 10:
        base = [
            "#e41a1c",  # rouge
            "#377eb8",  # bleu
            "#4daf4a",  # vert
            "#984ea3",  # violet
            "#ff7f00",  # orange
            "#ffff33",  # jaune
            "#a65628",  # brun
            "#f781bf",  # rose
            "#000000",  # noir
            "#00ced1",  # turquoise
        ]
        return base[:n]

    # au-delà de 10, on échantillonne un colormap catégoriel
    cmap = plt.get_cmap("gist_ncar", n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


