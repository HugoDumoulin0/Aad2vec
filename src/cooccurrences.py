#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:49:57 2026

@author: hugodumoulin
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import AgglomerativeClustering

from loaders import available_labels, load_sentences
from pos_utils import get_word_pos
from utils import normalize_rows, sentence_tokens


COOC_CACHE = {}
COOC_MATRIX_CACHE = {}


def build_cooc_matrix(sentences, vocab, window_size=4):
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    rows, cols, data = [], [], []
    counts = defaultdict(float)

    for sent in sentences:
        toks = [t for t in sentence_tokens(sent) if t in word_to_idx]

        for i, w in enumerate(toks):
            wi = word_to_idx[w]

            start = max(0, i - window_size)
            end = min(len(toks), i + window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue
                cj = word_to_idx[toks[j]]
                counts[(wi, cj)] += 1.0

    for (r, c), v in counts.items():
        rows.append(r)
        cols.append(c)
        data.append(v)

    M = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(vocab), len(vocab)),
        dtype=np.float32
    )
       
    return M


def _notify(status_callback, message):
    if status_callback is not None:
        status_callback(message)


def build_full_ppmi_matrix(
    sentences,
    window_size=5,
    min_count=1,
    cache_key=None,
    status_callback=None,
):
    if cache_key is not None and cache_key in COOC_MATRIX_CACHE:
        _notify(status_callback, "Matrice PPMI complète récupérée depuis le cache.")
        return COOC_MATRIX_CACHE[cache_key]

    _notify(status_callback, "Construction du vocabulaire complet...")
    all_tokens = []
    for sent in sentences:
        all_tokens.extend(sentence_tokens(sent))

    counts = Counter(all_tokens)
    min_count = max(1, int(min_count))
    full_vocab = [
        w for w, count in counts.most_common()
        if count >= min_count
    ]

    if len(full_vocab) < 5:
        result = (full_vocab, sp.csr_matrix((0, 0), dtype=np.float32))
        if cache_key is not None:
            COOC_MATRIX_CACHE[cache_key] = result
        return result

    _notify(status_callback, f"Construction de la matrice de cooccurrence ({len(full_vocab)} mots)...")
    counts_matrix = build_cooc_matrix(
        sentences=sentences,
        vocab=full_vocab,
        window_size=window_size,
    )

    if counts_matrix.nnz == 0:
        result = (full_vocab, sp.csr_matrix(counts_matrix.shape, dtype=np.float32))
        if cache_key is not None:
            COOC_MATRIX_CACHE[cache_key] = result
        return result

    _notify(status_callback, "Calcul de la matrice PPMI complète...")
    coo = counts_matrix.tocoo()
    total = float(counts_matrix.sum())
    row_sum = np.asarray(counts_matrix.sum(axis=1)).ravel()
    col_sum = np.asarray(counts_matrix.sum(axis=0)).ravel()

    denominator = row_sum[coo.row] * col_sum[coo.col]
    valid = denominator > 0

    ppmi_data = np.zeros_like(coo.data, dtype=np.float32)
    ppmi_data[valid] = np.log((coo.data[valid].astype(float) * total) / denominator[valid])
    ppmi_data[ppmi_data < 0] = 0.0

    positive = ppmi_data > 0
    ppmi = sp.csr_matrix(
        (ppmi_data[positive], (coo.row[positive], coo.col[positive])),
        shape=counts_matrix.shape,
        dtype=np.float32,
    )

    result = (full_vocab, ppmi)
    if cache_key is not None:
        COOC_MATRIX_CACHE[cache_key] = result

    return result


def top_cooccurring_words(sentences, target_word, window_size=4, topn=20, min_frequency=1):
    if not sentences or not target_word:
        return pd.DataFrame(columns=["rank", "word", "ppmi", "cooccurrences"])

    target_word = str(target_word)
    min_frequency = max(1, int(min_frequency))
    token_counts = Counter()
    pair_counts = Counter()
    row_totals = Counter()
    col_totals = Counter()
    total_pairs = 0

    for sent in sentences:
        toks = [str(t) for t in sentence_tokens(sent)]
        token_counts.update(toks)

        for i, center_word in enumerate(toks):
            start = max(0, i - int(window_size))
            end = min(len(toks), i + int(window_size) + 1)

            for j in range(start, end):
                if i == j:
                    continue

                context_word = toks[j]
                pair_counts[(center_word, context_word)] += 1
                row_totals[center_word] += 1
                col_totals[context_word] += 1
                total_pairs += 1

    if total_pairs == 0 or row_totals[target_word] == 0:
        return pd.DataFrame(columns=["rank", "word", "ppmi", "cooccurrences"])

    rows = []
    target_total = float(row_totals[target_word])

    for (center_word, word), count in pair_counts.items():
        if center_word != target_word or word == target_word:
            continue

        if token_counts[word] < min_frequency:
            continue

        denominator = target_total * float(col_totals[word])
        if denominator == 0.0:
            ppmi = 0.0
        else:
            pmi = np.log((float(count) * float(total_pairs)) / denominator)
            ppmi = max(float(pmi), 0.0)

        rows.append({
            "word": word,
            "ppmi": round(ppmi, 4),
            "cooccurrences": int(count),
        })

    rows = sorted(
        rows,
        key=lambda row: (row["ppmi"], row["cooccurrences"], row["word"]),
        reverse=True,
    )[:int(topn)]

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return pd.DataFrame(rows, columns=["rank", "word", "ppmi", "cooccurrences"])


def _token_counts(sentences):
    counts = Counter()
    for sent in sentences:
        counts.update(str(t) for t in sentence_tokens(sent))
    return counts


def _cooc_count_stats(sentences, window_size=4):
    pair_counts = Counter()
    row_totals = Counter()
    col_totals = Counter()
    total_pairs = 0

    for sent in sentences:
        toks = [str(t) for t in sentence_tokens(sent)]

        for i, center_word in enumerate(toks):
            start = max(0, i - int(window_size))
            end = min(len(toks), i + int(window_size) + 1)

            for j in range(start, end):
                if i == j:
                    continue

                context_word = toks[j]
                pair_counts[(center_word, context_word)] += 1
                row_totals[center_word] += 1
                col_totals[context_word] += 1
                total_pairs += 1

    return pair_counts, row_totals, col_totals, total_pairs


def _ppmi_value(count, row_total, col_total, total_pairs):
    denominator = float(row_total) * float(col_total)
    if denominator == 0.0 or total_pairs == 0:
        return 0.0

    pmi = np.log((float(count) * float(total_pairs)) / denominator)
    return max(float(pmi), 0.0)


def _ppmi_rows_from_counts(pair_counts, row_totals, col_totals, total_pairs):
    rows = defaultdict(dict)

    for (center_word, context_word), count in pair_counts.items():
        ppmi = _ppmi_value(
            count=count,
            row_total=row_totals[center_word],
            col_total=col_totals[context_word],
            total_pairs=total_pairs,
        )
        if ppmi > 0.0:
            rows[center_word][context_word] = ppmi

    return rows


def top_similar_words_by_cooc(sentences, target_word, window_size=4, topn=20, min_frequency=1):
    if not sentences or not target_word:
        return pd.DataFrame(columns=["rank", "word", "similarity", "frequency"])

    target_word = str(target_word)
    min_frequency = max(1, int(min_frequency))
    token_counts = _token_counts(sentences)
    pair_counts, row_totals, col_totals, total_pairs = _cooc_count_stats(
        sentences=sentences,
        window_size=window_size,
    )

    if total_pairs == 0 or target_word not in row_totals:
        return pd.DataFrame(columns=["rank", "word", "similarity", "frequency"])

    ppmi_rows = _ppmi_rows_from_counts(
        pair_counts=pair_counts,
        row_totals=row_totals,
        col_totals=col_totals,
        total_pairs=total_pairs,
    )
    target_row = ppmi_rows.get(target_word, {})

    if not target_row:
        return pd.DataFrame(columns=["rank", "word", "similarity", "frequency"])

    target_norm = np.sqrt(sum(v * v for v in target_row.values()))
    if target_norm == 0.0:
        return pd.DataFrame(columns=["rank", "word", "similarity", "frequency"])

    rows = []

    for word, row in ppmi_rows.items():
        if word == target_word or not row:
            continue

        if token_counts[word] < min_frequency:
            continue

        row_norm = np.sqrt(sum(v * v for v in row.values()))
        if row_norm == 0.0:
            continue

        common_contexts = set(target_row) & set(row)
        if not common_contexts:
            continue

        dot = sum(target_row[ctx] * row[ctx] for ctx in common_contexts)
        similarity = float(dot / (target_norm * row_norm))

        rows.append({
            "word": word,
            "similarity": round(similarity, 4),
            "frequency": int(token_counts[word]),
        })

    rows = sorted(
        rows,
        key=lambda row: (row["similarity"], row["word"]),
        reverse=True,
    )[:int(topn)]

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return pd.DataFrame(rows, columns=["rank", "word", "similarity", "frequency"])


def characteristic_contexts_for_word_cooc(
    sentences,
    target_word,
    window_size=4,
    topn_words=20,
    topn_contexts=10,
    min_frequency=1,
):
    cooc_df = top_cooccurring_words(
        sentences=sentences,
        target_word=target_word,
        window_size=window_size,
        topn=topn_words,
        min_frequency=min_frequency,
    )

    columns = [
        "rank",
        "score",
        "n_matched",
        "matched_words",
        "context",
        "sentence_id",
        "start",
        "end",
    ]

    if cooc_df.empty:
        return pd.DataFrame(columns=columns)

    ppmi_by_word = {
        str(row["word"]): float(row["ppmi"])
        for _, row in cooc_df.iterrows()
    }
    cooc_words = set(ppmi_by_word)
    target_word = str(target_word)
    context_window_size = int(window_size) * 2

    rows = []

    for sent_id, sent in enumerate(sentences):
        toks = [str(t) for t in sentence_tokens(sent)]
        word_toks = [str(t) for t in sent.get("word", toks)] if isinstance(sent, dict) else toks

        if len(word_toks) != len(toks):
            word_toks = toks

        for i, tok in enumerate(toks):
            if tok != target_word:
                continue

            start = max(0, i - context_window_size)
            end = min(len(toks), i + context_window_size + 1)
            window_tokens = toks[start:end]
            window_words = word_toks[start:end]

            matched = [
                w for j, w in enumerate(window_tokens)
                if start + j != i and w in cooc_words
            ]

            if not matched:
                continue

            score = sum(ppmi_by_word[w] for w in matched)
            matched_unique = sorted(set(matched), key=lambda w: ppmi_by_word[w], reverse=True)

            rows.append({
                "score": round(float(score), 4),
                "n_matched": int(len(matched)),
                "matched_words": ", ".join(matched_unique),
                "context": " ".join(window_words),
                "sentence_id": int(sent_id),
                "start": int(start),
                "end": int(end - 1),
            })

    if not rows:
        return pd.DataFrame(columns=columns)

    rows = sorted(
        rows,
        key=lambda row: (row["score"], row["n_matched"], row["context"]),
        reverse=True,
    )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["context"], keep="first").head(int(topn_contexts))
    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    return df[columns]


def compute_global_cooc_pca(run_name, allowed_words=None, n_components=100, window_size=4):
    labels = available_labels(run_name)

    all_sentences = []
    for label in labels:
        s = load_sentences(run_name, label)
        if s:
            all_sentences.extend(s)

    if not all_sentences:
        return pd.DataFrame(), {}, None, None, None

    if allowed_words is not None:
        vocab = sorted(list(set(allowed_words)))
    else:
        # construire le vocab depuis les tokens des phrases
        all_tokens = []
        for sent in all_sentences:
            tokens = sent["tokens"] if isinstance(sent, dict) else sent
            all_tokens.extend(tokens)
        counts = Counter(all_tokens)
        vocab = [w for w, _ in counts.most_common(1000)]  # ou un paramètre top_n

    M = build_cooc_matrix(
        all_sentences,
        vocab=vocab,
        window_size=window_size
    )

    svd = TruncatedSVD(n_components=min(n_components, len(vocab)-1), random_state=0)
    X = svd.fit_transform(M)

    X = normalize_rows(X)

    pca = PCA(n_components=2, random_state=0)
    XY = pca.fit_transform(X)

    df = pd.DataFrame({
        "word": vocab,
        "x": XY[:, 0],
        "y": XY[:, 1],
    })

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_words": len(vocab),
    }

    global_vectors = dict(zip(vocab, X))

    return df, meta, pca, global_vectors, svd


def compute_cooc_pca(
    sentences,
    allowed_pos=("ALL",),
    top_n=500,
    window_size=5,
    n_clusters=8,
    min_count=1,
    cluster_on_pca=True,
    matrix_cache_key=None,
    status_callback=None,
):
    if not sentences:
        return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}

    allowed_pos = set(allowed_pos) if allowed_pos else None
    if allowed_pos and "ALL" in allowed_pos:
        allowed_pos = None

    full_vocab, ppmi = build_full_ppmi_matrix(
        sentences=sentences,
        window_size=window_size,
        min_count=min_count,
        cache_key=matrix_cache_key,
        status_callback=status_callback,
    )

    if len(full_vocab) < 5:
        return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}

    vocab_index = {w: i for i, w in enumerate(full_vocab)}
    if ppmi.nnz == 0:
        return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}

    _notify(status_callback, "Sélection des mots affichés...")
    selected_words = full_vocab
    if allowed_pos:
        selected_words = [
            w for w in full_vocab
            if get_word_pos(w) in allowed_pos
        ]

    if top_n is not None:
        selected_words = selected_words[:int(top_n)]

    if len(selected_words) < 5:
        return pd.DataFrame(columns=["word", "x", "y", "cluster"]), {}

    selected_indices = [vocab_index[w] for w in selected_words]
    ppmi_selected = ppmi[selected_indices, :]

    _notify(status_callback, f"Réduction SVD + ACP sur {len(selected_words)} mots...")
    dim = min(100, max(2, min(ppmi_selected.shape) - 1))
    svd = TruncatedSVD(n_components=dim, random_state=42)
    X = svd.fit_transform(ppmi_selected)
    X = normalize_rows(X)

    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)

    n_clusters = max(2, min(int(n_clusters), len(selected_words) - 1))
    X_cluster = XY if cluster_on_pca else X

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="ward"
    )

    cluster_labels = clustering.fit_predict(X_cluster)

    df = pd.DataFrame({
        "word": selected_words,
        "x": XY[:, 0],
        "y": XY[:, 1],
        "cluster": cluster_labels,
    })

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_words": len(selected_words),
        "n_vocab_total": len(full_vocab),
        "n_clusters": n_clusters,
        "min_count": min_count,
    }

    return df, meta



def project_word_across_subcorpora_cooc(
    run_name,
    labels,
    allowed_words,
    global_vectors,
    pca,
    svd,
    window_size=4
):
    rows = []

    vocab = sorted(list(global_vectors.keys()))
    X_global = np.vstack([global_vectors[w] for w in vocab])

    for label in labels:
        sentences = load_sentences(run_name, label)

        if not sentences:
            continue

        M_sub = build_cooc_matrix(
            sentences,
            vocab=vocab,
            window_size=window_size
        )

        if M_sub.shape[0] < 2:
            continue

        try:
            X_sub = svd.transform(M_sub)
        except Exception:
            continue

        X_sub = normalize_rows(X_sub)

        XY = pca.transform(X_sub)

        cos_sims = np.sum(X_sub * X_global, axis=1)
        cos_dists = 1.0 - cos_sims

        for i, w in enumerate(vocab):
            rows.append({
                "subcorpus": label,
                "word": w,
                "x": XY[i, 0],
                "y": XY[i, 1],
                "cosine_similarity_to_global": float(cos_sims[i]),
                "cosine_distance_to_global": float(cos_dists[i]),
            })

    return pd.DataFrame(rows)
