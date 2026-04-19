#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:03:42 2026

@author: hugodumoulin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from shiny import App, ui, render, reactive
import os
import json
from functools import lru_cache

import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.linalg import orthogonal_procrustes

import hashlib

OUTPUT_DIR = "../outputs"
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")


# ============================================================
# Cache
# ============================================================
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

# ============================================================
# Sauver et charger
# ============================================================

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


def stable_short_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

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

# ============================================================
# Utilitaires chargement
# ============================================================
def available_runs():
    if not os.path.isdir(RUNS_DIR):
        return []

    runs = []
    for name in os.listdir(RUNS_DIR):
        path = os.path.join(RUNS_DIR, name)
        if os.path.isdir(path):
            runs.append(name)

    return sorted(runs, reverse=True)


def run_dirs(run_name):
    if not run_name:
        return {}

    base = os.path.join(RUNS_DIR, run_name)
    return {
        "run_dir": base,
        "models": os.path.join(base, "models"),
        "sentences": os.path.join(base, "sentences"),
        "semantic_shift": os.path.join(base, "semantic_shift"),
        "metadata": os.path.join(base, "metadata.json"),
    }


def load_run_metadata(run_name):
    dirs = run_dirs(run_name)
    meta_path = dirs.get("metadata")

    if not meta_path or not os.path.exists(meta_path):
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def available_labels(run_name):
    dirs = run_dirs(run_name)
    models_dir = dirs.get("models")

    if not models_dir or not os.path.isdir(models_dir):
        return []

    labels = []
    for f in os.listdir(models_dir):
        if f.startswith("w2v_") and f.endswith(".model") and f != "w2v_global.model":
            labels.append(f[len("w2v_"):-len(".model")])

    return sorted(labels)

def load_sentences(run_name, label):
    """
    Supporte :
    - list[list[str]]
    - list[dict] avec clé 'tokens'
    """
    dirs = run_dirs(run_name)
    path = os.path.join(dirs["sentences"], f"{label}_sentences.json")
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return []

    # cas 1 : déjà une liste de listes de tokens
    if isinstance(data, list) and isinstance(data[0], list):
        return data

    # cas 2 : liste de dicts avec clé tokens
    if isinstance(data, list) and isinstance(data[0], dict):
        out = []
        for item in data:
            tokens = item.get("tokens", [])
            if isinstance(tokens, list):
                out.append(tokens)
        return out

    return []

def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


@lru_cache(maxsize=256)
def load_model(run_name, label):
    dirs = run_dirs(run_name)
    path = os.path.join(dirs["models"], f"w2v_{label}.model")
    if not os.path.exists(path):
        return None
    return Word2Vec.load(path)


@lru_cache(maxsize=64)
def load_global_model(run_name):
    dirs = run_dirs(run_name)
    path = os.path.join(dirs["models"], "w2v_global.model")
    if not os.path.exists(path):
        return None
    return Word2Vec.load(path)


# ============================================================
# palette dynamique pour shift
# ============================================================
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

# ============================================================
# Vectoriel
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
# Extraction robuste de fenêtres minimales
# ============================================================

def select_non_overlapping_windows(df, topn=50):
    if df.empty:
        return df

    selected = []
    occupied = {}

    for _, row in df.iterrows():
        sent_id = int(row["sentence_id"])
        start = int(row["start"])
        end = int(row["end"])

        if sent_id not in occupied:
            occupied[sent_id] = []
            selected.append(row)
            occupied[sent_id].append((start, end))
            if len(selected) >= topn:
                break
            continue

        overlap = False
        for s, e in occupied[sent_id]:
            if not (end < s or start > e):
                overlap = True
                break

        if not overlap:
            selected.append(row)
            occupied[sent_id].append((start, end))
            if len(selected) >= topn:
                break

    return pd.DataFrame(selected).reset_index(drop=True)

def extract_cluster_windows(
    sentences,
    cluster_words,
    model,
    window_size=5,
    topn=50,
    min_unique_cluster_words=1,
    deduplicate=True,
    use_cluster_bonus=True,
    cluster_bonus_weight=0.20,
):
    """
    Extrait des fenêtres locales centrées sur les occurrences des mots du cluster.
    Score principal :
        cosine(window_vec, cluster_centroid)
    Bonus :
        nombre de mots distincts du cluster présents dans la fenêtre
    """

    if model is None:
        return pd.DataFrame(columns=[
            "sentence_id",
            "center_word",
            "window_text",
            "matched_cluster_words",
            "n_matched",
            "sim_cluster",
            "score",
        ])

    cluster_words = [w for w in cluster_words if w in model.wv]
    if not cluster_words:
        return pd.DataFrame(columns=[
            "sentence_id",
            "center_word",
            "window_text",
            "matched_cluster_words",
            "n_matched",
            "sim_cluster",
            "score",
        ])

    cluster_set = set(cluster_words)
    centroid = cluster_centroid(cluster_words, model)
    if centroid is None:
        return pd.DataFrame(columns=[
            "sentence_id",
            "center_word",
            "window_text",
            "matched_cluster_words",
            "n_matched",
            "sim_cluster",
            "score",
        ])

    rows = []

    for sent_id, sent in enumerate(sentences):
        if not sent:
            continue

        # normalisation légère : on garde les tokens tels quels,
        # puisqu'ils doivent matcher le vocabulaire appris
        tokens = [str(t) for t in sent]

        for i, tok in enumerate(tokens):
            if tok not in cluster_set:
                continue

            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            win_tokens = tokens[start:end]

            matched = sorted(set([w for w in win_tokens if w in cluster_set]))
            n_matched = len(matched)

            if n_matched < min_unique_cluster_words:
                continue

            win_vec = mean_vector(win_tokens, model)
            if win_vec is None:
                continue

            sim_cluster = cosine(win_vec, centroid)

            # petit bonus si plusieurs mots du cluster coexistent dans la fenêtre
            if use_cluster_bonus:
                score = sim_cluster + cluster_bonus_weight * max(0, n_matched - 1)
            else:
                score = sim_cluster

            rows.append({
                "sentence_id": sent_id,
                "start": start,
                "end": end - 1,
                "center_word": tok,
                "window_text": " ".join(win_tokens),
                "matched_cluster_words": ", ".join(matched),
                "n_matched": n_matched,
                "sim_cluster": round(sim_cluster, 4),
                "score": round(score, 4),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(
        ["score", "sim_cluster", "n_matched"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    if deduplicate:
        df = df.drop_duplicates(subset=["window_text"], keep="first").reset_index(drop=True)
    
    df = select_non_overlapping_windows(df, topn=topn)

    return df

def get_context_vectors(model):
    """
    Retourne la matrice des vecteurs de contexte et le vocabulaire aligné.
    Compatible negative sampling (syn1neg) et hierarchical softmax (syn1).
    """
    if model is None:
        return None, []

    vocab = list(model.wv.index_to_key)

    if hasattr(model, "syn1neg") and model.syn1neg is not None and len(model.syn1neg) == len(vocab):
        return model.syn1neg, vocab

    if hasattr(model, "syn1") and model.syn1 is not None and len(model.syn1) == len(vocab):
        return model.syn1, vocab

    return None, []

def characteristic_context_words_for_cluster(
    cluster_words,
    model,
    topn=30,
    min_count=1,
    exclude_cluster_words=True,
    allowed_pos=None,
    use_cosine=True,
):
    """
    Retourne les mots de contexte les plus associés à un cluster,
    en utilisant les vecteurs c de contexte.
    """
    if model is None:
        return pd.DataFrame(columns=["context_word", "score", "frequency"])

    cluster_words = [w for w in cluster_words if w in model.wv]
    if not cluster_words:
        return pd.DataFrame(columns=["context_word", "score", "frequency"])

    centroid = cluster_centroid(cluster_words, model)
    if centroid is None:
        return pd.DataFrame(columns=["context_word", "score", "frequency"])

    C, vocab = get_context_vectors(model)
    if C is None or len(vocab) == 0:
        return pd.DataFrame(columns=["context_word", "score", "frequency"])

    centroid = centroid.reshape(1, -1)

    if use_cosine:
        centroid_n = normalize_rows(centroid)[0]
        C_n = normalize_rows(C)
        scores = C_n @ centroid_n
    else:
        scores = C @ centroid[0]

    rows = []
    cluster_set = set(cluster_words)
    allowed_pos = set(allowed_pos) if allowed_pos else None

    for i, word in enumerate(vocab):
        freq = int(model.wv.get_vecattr(word, "count"))

        if freq < min_count:
            continue

        if exclude_cluster_words and word in cluster_set:
            continue

        if allowed_pos is not None and get_word_pos(word) not in allowed_pos:
            continue

        rows.append({
            "context_word": word,
            "score": float(scores[i]),
            "frequency": freq,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["score", "frequency"], ascending=[False, False]).reset_index(drop=True)
    df["score"] = df["score"].round(4)

    return df.head(topn)

def extract_sentences_from_characteristic_context_words(
    sentences,
    context_words_df,
    cluster_words=None,
    topn=50,
    require_cluster_word=False,
    min_context_matches=1,
    deduplicate=True,
):
    """
    Retrouve les phrases du corpus contenant les mots de contexte
    les plus associés au cluster.
    """
    if not sentences or context_words_df is None or context_words_df.empty:
        return pd.DataFrame(columns=[
            "sentence_id",
            "sentence_text",
            "matched_context_words",
            "n_context_matched",
            "matched_cluster_words",
            "n_cluster_matched",
            "score",
        ])

    context_words = context_words_df["context_word"].astype(str).tolist()
    context_score_map = {
        row["context_word"]: float(row["score"])
        for _, row in context_words_df.iterrows()
    }

    context_set = set(context_words)
    cluster_set = set(cluster_words) if cluster_words else set()

    rows = []

    for sent_id, sent in enumerate(sentences):
        if not sent:
            continue

        tokens = [str(t) for t in sent]

        matched_context = sorted(set([t for t in tokens if t in context_set]))
        n_context = len(matched_context)

        if n_context < min_context_matches:
            continue

        matched_cluster = sorted(set([t for t in tokens if t in cluster_set]))
        n_cluster = len(matched_cluster)

        if require_cluster_word and n_cluster == 0:
            continue

        score = sum(context_score_map[w] for w in matched_context)

        rows.append({
            "sentence_id": sent_id,
            "sentence_text": " ".join(tokens),
            "matched_context_words": ", ".join(matched_context),
            "n_context_matched": n_context,
            "matched_cluster_words": ", ".join(matched_cluster),
            "n_cluster_matched": n_cluster,
            "score": round(score, 4),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        ["score", "n_context_matched", "n_cluster_matched"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    if deduplicate:
        df = df.drop_duplicates(subset=["sentence_text"], keep="first").reset_index(drop=True)

    return df.head(topn)

def extract_windows_from_characteristic_context_words(
    sentences,
    context_words_df,
    cluster_words=None,
    window_size=5,
    topn=50,
    require_cluster_word=True,
    min_context_matches=1,
    deduplicate=True,
    non_overlapping=True,
):
    """
    Extrait des fenêtres locales centrées sur les mots de contexte
    les plus caractéristiques d'un cluster.

    Score =
        somme des scores des mots de contexte caractéristiques présents
        + bonus léger pour les mots du cluster présents

    Paramètres
    ----------
    sentences : list[list[str]]
        Corpus tokenisé.
    context_words_df : DataFrame
        Sortie de characteristic_context_words_for_cluster()
        avec colonnes: context_word, score, frequency.
    cluster_words : list[str] | None
        Mots du cluster cible.
    window_size : int
        Nombre de tokens à gauche et à droite du centre.
    topn : int
        Nombre max de fenêtres retournées.
    require_cluster_word : bool
        Si True, impose au moins un mot du cluster dans la fenêtre.
    min_context_matches : int
        Nombre minimum de mots de contexte caractéristiques présents.
    deduplicate : bool
        Déduplique les fenêtres identiques.
    non_overlapping : bool
        Si True, supprime les chevauchements intra-phrase.
    """
    empty_cols = [
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
    ]

    if not sentences or context_words_df is None or context_words_df.empty:
        return pd.DataFrame(columns=empty_cols)

    context_words = context_words_df["context_word"].dropna().astype(str).tolist()
    context_set = set(context_words)

    context_score_map = {
        str(row["context_word"]): float(row["score"])
        for _, row in context_words_df.iterrows()
    }

    cluster_set = set(cluster_words) if cluster_words else set()

    rows = []

    for sent_id, sent in enumerate(sentences):
        if not sent:
            continue

        tokens = [str(t) for t in sent]

        for i, tok in enumerate(tokens):
            # on centre sur un mot de contexte caractéristique
            if tok not in context_set:
                continue

            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            win_tokens = tokens[start:end]

            matched_context = sorted(set([w for w in win_tokens if w in context_set]))
            n_context = len(matched_context)

            if n_context < min_context_matches:
                continue

            matched_cluster = sorted(set([w for w in win_tokens if w in cluster_set]))
            n_cluster = len(matched_cluster)

            if require_cluster_word and n_cluster == 0:
                continue

            # score principal = somme des scores des mots de contexte présents
            score_context = sum(context_score_map[w] for w in matched_context)

            # petit bonus pour la présence de mots du cluster
            score = score_context + 0.1 * n_cluster

            rows.append({
                "sentence_id": sent_id,
                "start": start,
                "end": end - 1,
                "center_word": tok,
                "window_text": " ".join(win_tokens),
                "matched_context_words": ", ".join(matched_context),
                "n_context_matched": n_context,
                "matched_cluster_words": ", ".join(matched_cluster),
                "n_cluster_matched": n_cluster,
                "score": round(score, 4),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    df = df.sort_values(
        ["score", "n_context_matched", "n_cluster_matched"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    if deduplicate:
        df = df.drop_duplicates(subset=["window_text"], keep="first").reset_index(drop=True)

    if non_overlapping:
        df = select_non_overlapping_windows(df, topn=topn)
    else:
        df = df.head(topn).reset_index(drop=True)

    return df

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

POS_OPTIONS = [
    "NOUN",
    "PROPN",
    "ADJ",
    "VERB",
    "ADV",
    "ADP",
    "DET",
    "PRON",
    "AUX",
    "CCONJ",
    "SCONJ",
    "NUM",
    "INTJ",
]

@lru_cache(maxsize=1)
def get_nlp():
    return spacy.load("fr_core_news_sm")


@lru_cache(maxsize=50000)
def get_word_pos(word):
    nlp = get_nlp()
    doc = nlp(word)
    if len(doc) == 1:
        return doc[0].pos_
    return None


def get_words_sorted_by_frequency(model):
    return list(model.wv.index_to_key)


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
        words = df["word"].astype(str).tolist()
        if len(words) < 2:
            result = (df, meta, None)
            GLOBAL_PCA_CACHE[key] = result
            return result

        X = np.vstack([global_model.wv[w] for w in words if w in global_model.wv])
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


# ============================================================
# UI
# ============================================================

runs0 = available_runs()
env_default_run = os.environ.get("DEFAULT_RUN_NAME")
default_run = env_default_run if env_default_run in runs0 else (runs0[0] if runs0 else None)
labels0 = available_labels(default_run) if default_run else []

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
            )
        )
    )




# ============================================================
# Server
# ============================================================

def server(input, output, session):

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
            return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}

        if not label or label == "Aucune PCA disponible":
            return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}

        model = load_model(run_name, label)

        allowed_pos = tuple(input.pos_filter_sub()) if input.pos_filter_sub() is not None else tuple()
        top_n = int(input.top_words_sub())
        n_clusters = int(input.n_clusters_sub())

        df, meta = compute_dynamic_pca_and_clusters(
            run_name=run_name,
            model=model,
            label=label,
            allowed_pos=allowed_pos,
            top_n=top_n,
            n_clusters=n_clusters,
            cluster_on_pca=True
        )

        return df, meta
    
    
    @reactive.calc
    def global_pca_data():
        run_name = input.run_name()

        if not run_name or run_name == "Aucun run disponible":
            return pd.DataFrame(columns=["word", "x", "y"]), {}, None

        global_model = load_global_model(run_name)

        allowed_pos = tuple(input.pos_filter_global()) if input.pos_filter_global() is not None else tuple()
        top_n = int(input.top_words_global())

        df, meta, pca = compute_dynamic_global_pca(
            run_name=run_name,
            global_model=global_model,
            allowed_pos=allowed_pos,
            top_n=top_n
        )

        return df, meta, pca
    
    
    
    # ---- mise à jour labels lors du changement de run
    @reactive.Effect
    def _update_labels_for_run():
        run_name = input.run_name()
    
        labels = available_labels(run_name)
    
        if labels:
            current = input.corpus_label()
            selected = current if current in labels else labels[0]
            ui.update_select(
                "corpus_label",
                choices=labels,
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
        model = load_model(run_name, label)
        sentences = load_sentences(run_name, label)


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
        model = load_model(run_name, label)
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
        model = load_model(run_name, label)
    
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
        model = load_model(run_name, label)
        sentences = load_sentences(run_name, label)
    
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
        model = load_model(run_name, label)
        sentences = load_sentences(run_name, label)
    
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
            return df_global, meta, pd.DataFrame(columns=[
                "subcorpus", "word", "x", "y",
                "cosine_similarity_to_global",
                "cosine_distance_to_global"
            ])

        labels = available_labels(run_name)
        allowed_words = df_global["word"].dropna().astype(str).tolist()

        df_positions_all = project_all_words_across_subcorpora(
            pca=pca,
            global_model=global_model,
            labels=labels,
            allowed_words=allowed_words,
            run_name=run_name
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
            ]))
    
        top_n = int(input.top_spread_words())
        df_spread = df_spread.head(top_n).copy()
    
        # arrondis pour lisibilité
        df_spread["max_pairwise_dist_2d"] = df_spread["max_pairwise_dist_2d"].round(4)
        df_spread["mean_pairwise_dist_2d"] = df_spread["mean_pairwise_dist_2d"].round(4)
    
        return render.DataGrid(df_spread)
    
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
    
    
    # ---- info cache
    @output
    @render.text
    def cache_info():
        return (
            f"ACP sous-corpus en cache : {len(PCA_CACHE)}\n"
            f"ACP globales en cache : {len(GLOBAL_PCA_CACHE)}"
    )

app = App(app_ui, server)

