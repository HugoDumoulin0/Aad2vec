#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:41:58 2026

@author: hugodumoulin
"""

import os
import json
from functools import lru_cache
from gensim.models import Word2Vec

from config import RUNS_DIR, COMPLETE_CORPUS_LABEL


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
    Retourne une liste de dicts :

    {
        "tokens": [...],
        "word": [...],
        "lemma": [...],
        "surface_text": "..."
    }

    Compatible ancien format.
    """

    dirs = run_dirs(run_name)

    path = os.path.join(
        dirs["sentences"],
        f"{label}_sentences.json"
    )

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return []

    # ancien format
    if isinstance(data[0], list):

        return [
            {
                "tokens": sent,
                "word": sent,
                "lemma": sent,
                "surface_text": " ".join(sent),
            }
            for sent in data
        ]

    # nouveau format
    if isinstance(data[0], dict):

        out = []

        for item in data:

            tokens = item.get("tokens", [])

            word_tokens = item.get(
                "word",
                tokens
            )

            lemma_tokens = item.get(
                "lemma",
                tokens
            )

            surface_text = item.get(
                "surface_text",
                " ".join(word_tokens)
            )

            if isinstance(tokens, list):

                out.append({
                    "tokens": tokens,
                    "word": word_tokens,
                    "lemma": lemma_tokens,
                    "surface_text": surface_text,
                })

        return out

    return []

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

def corpus_choices(run_name):
    labels = available_labels(run_name)
    if not labels:
        return ["Aucune PCA disponible"]
    return [COMPLETE_CORPUS_LABEL] + labels


def load_model_for_corpus(run_name, label):
    if label == COMPLETE_CORPUS_LABEL:
        return load_global_model(run_name)
    return load_model(run_name, label)


def load_sentences_for_corpus(run_name, label):
    if label == COMPLETE_CORPUS_LABEL:
        sentences = []
        for sub_label in available_labels(run_name):
            sentences.extend(load_sentences(run_name, sub_label))
        return sentences

    return load_sentences(run_name, label)

def pca_cache_label(label):
    if label == COMPLETE_CORPUS_LABEL:
        return "GLOBAL"
    return label
