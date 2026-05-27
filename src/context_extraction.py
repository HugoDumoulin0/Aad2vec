#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:58:15 2026

@author: hugodumoulin
"""

import pandas as pd

from utils import (
    cosine,
    mean_vector,
    cluster_centroid,
    normalize_rows,
    sentence_tokens,
)
from pos_utils import get_word_pos



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
        tokens = [str(t) for t in sentence_tokens(sent)]


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

        tokens = [str(t) for t in sentence_tokens(sent)]


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

    context_words = context_words_df["context_word"].astype(str).tolist()
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

        # toujours la représentation du modèle
        tokens = sent["tokens"]

        # toujours la représentation lisible
        word_tokens = sent["word"]

        if not tokens:
            continue

        for i, tok in enumerate(tokens):

            if tok not in context_set:
                continue

            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)

            win_tokens = tokens[start:end]
            win_words = word_tokens[start:end]

            matched_context = list(
                set(w for w in win_tokens if w in context_set)
            )

            n_context = len(matched_context)

            if n_context < min_context_matches:
                continue

            matched_cluster = list(
                set(w for w in win_tokens if w in cluster_set)
            )

            n_cluster = len(matched_cluster)

            if require_cluster_word and n_cluster == 0:
                continue

            score_context = sum(
                context_score_map[w]
                for w in matched_context
            )

            score = score_context + 0.1 * n_cluster

            rows.append({
                "sentence_id": sent_id,
                "start": start,
                "end": end - 1,

                # centre côté modèle
                "center_word": tok,

                # affichage réel
                "window_text": " ".join(win_words),

                "matched_context_words":
                    ", ".join(sorted(matched_context)),

                "n_context_matched": n_context,

                "matched_cluster_words":
                    ", ".join(sorted(matched_cluster)),

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
        df = df.drop_duplicates(
            subset=["window_text"],
            keep="first"
        ).reset_index(drop=True)

    if non_overlapping:
        df = select_non_overlapping_windows(
            df,
            topn=topn
        )
    else:
        df = df.head(topn)

    return df