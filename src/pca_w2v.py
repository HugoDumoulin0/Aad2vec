#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:39:17 2026

@author: hugodumoulin
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from loaders import run_dirs
from pos_utils import get_word_pos
from utils import normalize_rows, stable_short_hash


PCA_CACHE = {}
GLOBAL_PCA_CACHE = {}

def make_sub_pca_cache_key(run_name, label, allowed_pos, top_n, n_clusters, cluster_on_pca=True):
    return (
        run_name,
        label,
        tuple(sorted(allowed_pos)) if allowed_pos else tuple(),
        int(top_n),
        int(n_clusters),
        bool(cluster_on_pca),
    )


def make_global_pca_cache_key(run_name, allowed_pos, top_n):
    return (
        run_name,
        tuple(sorted(allowed_pos)) if allowed_pos else tuple(),
        int(top_n),
    )

def compute_dynamic_pca_and_clusters(
    run_name,
    model,
    label,
    allowed_pos=None,
    top_n=400,
    n_clusters=8,
    cluster_on_pca=True
    ):
    
    key = make_sub_pca_cache_key(run_name, label, allowed_pos, top_n, n_clusters, cluster_on_pca)
    if key in PCA_CACHE:
        return PCA_CACHE[key]
    
    csv_path, meta_path = subcorpus_pca_filepaths(run_name, label, allowed_pos, top_n, n_clusters)

    if os.path.exists(csv_path) and os.path.exists(meta_path):
        df = pd.read_csv(csv_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        result = (df, meta)
        PCA_CACHE[key] = result
        return result

    if model is None:
        result = (pd.DataFrame(columns=["word", "x", "y", "cluster"]), {})
        PCA_CACHE[key] = result
        return result

    words = select_words_by_pos_then_topn(
        model=model,
        allowed_pos=allowed_pos,
        top_n=top_n
    )

    if len(words) < 2:
        result = (pd.DataFrame(columns=["word", "x", "y", "cluster"]), {})
        PCA_CACHE[key] = result
        return result

    X = np.vstack([model.wv[w] for w in words])

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    n_clusters = int(n_clusters)
    if len(words) <= 2:
        n_clusters = 2
    else:
        n_clusters = max(2, min(n_clusters, len(words) - 1))

    X_cluster = X_2d if cluster_on_pca else X

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward"
    )

    cluster_labels = clustering.fit_predict(X_cluster)

    df = pd.DataFrame({
        "word": words,
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "cluster": cluster_labels
    })

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_words": len(words),
        "n_clusters": n_clusters,
    }

    save_subcorpus_pca_csv(
        run_name=run_name,
        label=label,
        allowed_pos=allowed_pos,
        top_n=top_n,
        n_clusters=n_clusters,
        df=df,
        meta=meta,
    )
    
    result = (df, meta)
    PCA_CACHE[key] = result
    return result


def compute_dynamic_global_pca(run_name, global_model, allowed_pos=None, top_n=1000):
    key = make_global_pca_cache_key(run_name, allowed_pos, top_n)

    if key in GLOBAL_PCA_CACHE:
        return GLOBAL_PCA_CACHE[key]

    csv_path, meta_path = global_pca_filepaths(run_name, allowed_pos, top_n)

    if os.path.exists(csv_path) and os.path.exists(meta_path):
        df = pd.read_csv(csv_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        # on doit quand même reconstruire la PCA pour les projections ultérieures
        words = [
            w for w in df["word"].astype(str).tolist()
            if global_model is not None and w in global_model.wv
        ]
        
        if len(words) < 2:
            result = (df, meta, None)
            GLOBAL_PCA_CACHE[key] = result
            return result
        
        X = np.vstack([global_model.wv[w] for w in words])
        X = normalize_rows(X)


        if len(X) < 2:
            result = (df, meta, None)
            GLOBAL_PCA_CACHE[key] = result
            return result

        pca = PCA(n_components=2)
        pca.fit(X)

        result = (df, meta, pca)
        GLOBAL_PCA_CACHE[key] = result
        return result

    if global_model is None:
        result = (pd.DataFrame(columns=["word", "x", "y"]), {}, None)
        GLOBAL_PCA_CACHE[key] = result
        return result

    words = select_words_by_pos_then_topn(
        model=global_model,
        allowed_pos=allowed_pos,
        top_n=top_n
    )

    if len(words) < 2:
        result = (pd.DataFrame(columns=["word", "x", "y"]), {}, None)
        GLOBAL_PCA_CACHE[key] = result
        return result

    X = np.vstack([global_model.wv[w] for w in words])
    X = normalize_rows(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    df = pd.DataFrame({
        "word": words,
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
    })

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_words": len(words),
    }
    
    save_global_pca_csv(
    run_name=run_name,
    allowed_pos=allowed_pos,
    top_n=top_n,
    df=df,
    meta=meta,
    )

    result = (df, meta, pca)
    GLOBAL_PCA_CACHE[key] = result
    return result



def add_pca_contributions(df, meta=None):
    """
    Ajoute des colonnes de contribution à partir des coordonnées PCA.
    """
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        return df

    df = df.copy()

    sum_x2 = float((df["x"] ** 2).sum())
    sum_y2 = float((df["y"] ** 2).sum())

    if sum_x2 == 0:
        df["contrib_dim1"] = 0.0
    else:
        df["contrib_dim1"] = (df["x"] ** 2) / sum_x2

    if sum_y2 == 0:
        df["contrib_dim2"] = 0.0
    else:
        df["contrib_dim2"] = (df["y"] ** 2) / sum_y2

    evr = []
    if meta is not None:
        evr = meta.get("explained_variance_ratio", [])

    if len(evr) >= 2:
        w1, w2 = evr[0], evr[1]
    else:
        w1, w2 = 0.5, 0.5

    df["contrib_total"] = w1 * df["contrib_dim1"] + w2 * df["contrib_dim2"]

    return df


def get_words_sorted_by_frequency(model):
    return list(model.wv.index_to_key)


def filter_top_contributors(df, pct=100):
    if df.empty or "contrib_total" not in df.columns:
        return df

    pct = max(1, min(100, int(pct)))

    if pct == 100:
        return df

    n_keep = max(1, int(round(len(df) * pct / 100.0)))

    return (
        df.sort_values("contrib_total", ascending=False)
          .head(n_keep)
          .copy()
          )
    


def filter_words_by_pos(words, allowed_pos):
    if not words:
        return []

    if not allowed_pos:
        return words

    allowed_pos = set(allowed_pos)
    return [w for w in words if get_word_pos(w) in allowed_pos]


def select_words_by_pos_then_topn(model, allowed_pos=None, top_n=400):
    words = get_words_sorted_by_frequency(model)
    words = filter_words_by_pos(words, allowed_pos)

    if top_n is not None:
        words = words[:int(top_n)]

    return words

def ensure_derived_dirs(run_name):
    dirs = run_dirs(run_name)
    derived_dir = os.path.join(dirs["run_dir"], "derived")
    sub_pca_dir = os.path.join(derived_dir, "pca_subcorpora")
    global_pca_dir = os.path.join(derived_dir, "pca_global")

    os.makedirs(sub_pca_dir, exist_ok=True)
    os.makedirs(global_pca_dir, exist_ok=True)

    return {
        "derived": derived_dir,
        "sub_pca": sub_pca_dir,
        "global_pca": global_pca_dir,
    }

def make_pos_tag(allowed_pos):
    if not allowed_pos:
        return "ALL"
    return "-".join(sorted(allowed_pos))

def save_subcorpus_pca_csv(run_name, label, allowed_pos, top_n, n_clusters, df, meta):
    paths = ensure_derived_dirs(run_name)

    pos_tag = make_pos_tag(allowed_pos)
    key_text = f"{run_name}|{label}|{pos_tag}|{top_n}|{n_clusters}"
    suffix = stable_short_hash(key_text)

    csv_path = os.path.join(
        paths["sub_pca"],
        f"{label}__pos_{pos_tag}__top_{top_n}__k_{n_clusters}__{suffix}.csv"
    )
    meta_path = csv_path.replace(".csv", ".meta.json")

    df.to_csv(csv_path, index=False, encoding="utf-8")

    payload = {
        "run_name": run_name,
        "label": label,
        "allowed_pos": list(allowed_pos) if allowed_pos else [],
        "top_n": int(top_n),
        "n_clusters": int(n_clusters),
        **meta,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return csv_path


def save_global_pca_csv(run_name, allowed_pos, top_n, df, meta):
    paths = ensure_derived_dirs(run_name)

    pos_tag = make_pos_tag(allowed_pos)
    key_text = f"{run_name}|GLOBAL|{pos_tag}|{top_n}"
    suffix = stable_short_hash(key_text)

    csv_path = os.path.join(
        paths["global_pca"],
        f"global__pos_{pos_tag}__top_{top_n}__{suffix}.csv"
    )
    meta_path = csv_path.replace(".csv", ".meta.json")

    df.to_csv(csv_path, index=False, encoding="utf-8")

    payload = {
        "run_name": run_name,
        "allowed_pos": list(allowed_pos) if allowed_pos else [],
        "top_n": int(top_n),
        **meta,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return csv_path


def subcorpus_pca_filepaths(run_name, label, allowed_pos, top_n, n_clusters):
    paths = ensure_derived_dirs(run_name)
    pos_tag = make_pos_tag(allowed_pos)
    key_text = f"{run_name}|{label}|{pos_tag}|{top_n}|{n_clusters}"
    suffix = stable_short_hash(key_text)

    csv_path = os.path.join(
        paths["sub_pca"],
        f"{label}__pos_{pos_tag}__top_{top_n}__k_{n_clusters}__{suffix}.csv"
    )
    meta_path = csv_path.replace(".csv", ".meta.json")
    return csv_path, meta_path


def global_pca_filepaths(run_name, allowed_pos, top_n):
    paths = ensure_derived_dirs(run_name)
    pos_tag = make_pos_tag(allowed_pos)
    key_text = f"{run_name}|GLOBAL|{pos_tag}|{top_n}"
    suffix = stable_short_hash(key_text)

    csv_path = os.path.join(
        paths["global_pca"],
        f"global__pos_{pos_tag}__top_{top_n}__{suffix}.csv"
    )
    meta_path = csv_path.replace(".csv", ".meta.json")
    return csv_path, meta_path

















