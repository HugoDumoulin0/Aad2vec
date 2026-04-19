#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:37:01 2026

@author: hugodumoulin
"""


import os
import json
import time
import signal
import subprocess
import sys
import webbrowser

import spacy
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from scipy.linalg import orthogonal_procrustes

from datetime import datetime


CORPUS_PATH = "../corpus"
SUBCORPORA_JSON = "subcorpora.json"
OUTPUT_DIR = "../outputs"
PCA_DIR = os.path.join(OUTPUT_DIR, "pca_data")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
SHIFT_DIR = os.path.join(OUTPUT_DIR, "semantic_shift")
LAUNCHER_JSON = "launcher_choice.json"
SUBCORPORA_JSON = "subcorpora.json"



#############################################
# Launcher
#############################################

def load_launcher_choice(path_json=LAUNCHER_JSON):
    if not os.path.exists(path_json):
        return None

    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)
    
def run_launcher_and_wait(choice_json=LAUNCHER_JSON, app_file="shiny_start.py", port="7999"):
    kill_process_on_port(port)

    if os.path.exists(choice_json):
        os.remove(choice_json)

    print(f"Lancement de {app_file}...")

    process = subprocess.Popen(
        [sys.executable, "-m", "shiny", "run", app_file, "--port", str(port)]
    )

    time.sleep(2)

    if process.poll() is not None:
        print(f"[ERREUR] {app_file} ne s'est pas lancé correctement.")
        return None

    url = f"http://127.0.0.1:{port}"
    print(f"Ouverture du navigateur : {url}")
    webbrowser.open(url)

    try:
        while not os.path.exists(choice_json):
            print(f"En attente de {choice_json}...")
            time.sleep(2)
    finally:
        process.send_signal(signal.SIGINT)
        process.wait()

    return choice_json




#############################################
# Utilitaires dossiers
#############################################

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PCA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SHIFT_DIR, exist_ok=True)
    

def save_sentences(sentences, label_safe, sentences_dir):
    path = os.path.join(sentences_dir, f"{label_safe}_sentences.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print(f"Phrases sauvegardées : {path} ({len(sentences)} phrases)")
    
def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


#############################################
# Chargement sous-corpus depuis JSON
#############################################

def load_pipeline_config(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ancien format : dict simple de groupes
    if isinstance(data, dict) and "subcorpora" not in data:
        return {
            "subcorpora": data,
            "training_config": {}
        }

    return data


def build_subcorpora_from_files(path_json, base_path=CORPUS_PATH):
    config = load_pipeline_config(path_json)
    groups = config["subcorpora"]

    subcorpora = {}

    for label, files in groups.items():
        textes = []
        for fichier in files:
            chemin = os.path.join(base_path, fichier)
            if not os.path.exists(chemin):
                print(f"[WARN] Fichier introuvable ignoré : {chemin}")
                continue
            with open(chemin, "r", encoding="utf-8") as f:
                textes.append(f.read())
        subcorpora[label] = textes

    return subcorpora

def get_training_config(path_json):
    config = load_pipeline_config(path_json)
    return config.get("training_config", {})

import re
import unicodedata


def safe_label(label):
    # normalisation unicode
    label = unicodedata.normalize("NFKD", str(label))
    label = label.encode("ascii", "ignore").decode("ascii")

    # minuscules
    label = label.lower().strip()

    # remplace slash, espaces et ponctuation par underscore
    label = re.sub(r"[^\w\s-]", "_", label)
    label = re.sub(r"[\s/\\]+", "_", label)

    # évite les répétitions d'underscore
    label = re.sub(r"_+", "_", label)

    # enlève underscore début/fin
    label = label.strip("_")

    if not label:
        label = "groupe"

    return label


#############################################
# Tokenisation
#############################################

def tokenize(corpus_liste, lowercase=True):
    nlp = spacy.load("fr_core_news_sm")
    corpus_sentences = []

    for texte in corpus_liste:
        if lowercase:
            texte = texte.lower()

        doc = nlp(texte, disable=["ner"])

        for sent in doc.sents:
            tokens = []
            for token in sent:
                if token.is_space:
                    continue
                tok = token.text
                if lowercase:
                    tok = tok.lower()
                tokens.append(tok)

            if tokens:
                corpus_sentences.append(tokens)

    return corpus_sentences


#############################################
# Entraînement Word2Vec
#############################################

def train(corpus, model_path, training_config):
    if not os.path.exists(model_path):
        print(f"Entraînement du modèle : {model_path}")

        vector_size = int(training_config.get("vector_size", 300))
        window = int(training_config.get("window", 2))
        min_count = int(training_config.get("min_count", 1))
        sg = int(training_config.get("sg", 1))
        negative = int(training_config.get("negative", 5))
        sample = float(training_config.get("sample", 0.001))
        workers = int(training_config.get("workers", 4))
        epochs = int(training_config.get("epochs", 50))

        w2v_model = Word2Vec(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            negative=negative,
            sample=sample,
        )

        start_time = time.time()

        for epoch in range(epochs):
            w2v_model.train(
                corpus,
                total_examples=w2v_model.corpus_count,
                epochs=1,
                compute_loss=True
            )
            loss = w2v_model.get_latest_training_loss()
            elapsed = time.time() - start_time
            print(
                f"{os.path.basename(model_path)} | Epoch {epoch+1}/{epochs} | "
                f"Loss: {loss:.2f} | Temps: {elapsed:.2f}s"
            )

        w2v_model.save(model_path)
    else:
        print(f"Chargement du modèle existant : {model_path}")
        w2v_model = Word2Vec.load(model_path)

    return w2v_model

#############################################
# Extraction vocabulaire brut
#############################################

def get_words_sorted_by_frequency(model):
    """
    Retourne le vocabulaire trié par fréquence décroissante.
    """
    return list(model.wv.index_to_key)


#############################################
# Pipeline principal NLP
#############################################

def process_subcorpora(path_json):
    ensure_dirs()

    config = load_pipeline_config(path_json)
    training_config = config.get("training_config", {})
    subcorpora = build_subcorpora_from_files(path_json)

    run_name = make_run_name(training_config)
    run_dirs = make_run_dirs(run_name)

    sub_models_dict = {}

    lowercase = bool(training_config.get("lowercase", False))

    for label, textes in subcorpora.items():
        print(f"\n=== Traitement du corpus : {label} ===")

        sentences = tokenize(textes, lowercase=lowercase)
        if not sentences:
            print("[WARN] Corpus vide après tokenisation.")
            continue

        label_safe = safe_label(label)

        save_sentences(sentences, label_safe, run_dirs["sentences"])

        model_path = os.path.join(run_dirs["models"], f"w2v_{label_safe}.model")
        model = train(sentences, model_path, training_config)
        sub_models_dict[label_safe] = model

    all_texts = []
    for textes in subcorpora.values():
        all_texts.extend(textes)

    global_sentences = tokenize(all_texts, lowercase=lowercase)
    global_model_path = os.path.join(run_dirs["models"], "w2v_global.model")
    global_model = train(global_sentences, global_model_path, training_config)

    build_semantic_shift_outputs(
        global_model=global_model,
        sub_models_dict=sub_models_dict,
        shift_dir=run_dirs["semantic_shift"],
        top_words=1000
    )

    save_run_metadata(
        run_dir=run_dirs["run_dir"],
        training_config=training_config,
        subcorpora_labels=list(sub_models_dict.keys())
    )

    return run_name

#############################################
# Alignement Procrustes (Hamilton et al 2016)
#############################################

def align_submodel_to_global(global_model, sub_model, anchor_topn=5000, min_anchor=200):
    """
    Aligne un modèle de sous-corpus sur le modèle global par orthogonal Procrustes.
    Retourne la matrice de rotation R telle que X_sub @ R ≈ X_global.
    """

    global_vocab = set(global_model.wv.key_to_index.keys())
    sub_vocab = set(sub_model.wv.key_to_index.keys())

    common = list(global_vocab & sub_vocab)

    # mots ancres = mots communs les plus fréquents dans le global
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

#############################################
# Export des positions alignées dans une ACP globale
#############################################

def build_semantic_shift_outputs(global_model, sub_models_dict, shift_dir, top_words=1000):
    """
    Construit les sorties pour l'onglet :
    - ACP globale de référence
    - positions alignées des mots par sous-corpus
    - distances cosinus au global
    """

    os.makedirs(shift_dir, exist_ok=True)

    # vocabulaire global de fond
    global_words = global_model.wv.index_to_key[:top_words]
    X_global = np.vstack([global_model.wv[w] for w in global_words])
    X_global = normalize_rows(X_global)

    pca = PCA(n_components=2)
    X_global_2d = pca.fit_transform(X_global)

    df_global = pd.DataFrame({
        "word": global_words,
        "x": X_global_2d[:, 0],
        "y": X_global_2d[:, 1],
    })
    df_global.to_csv(
        os.path.join(shift_dir, "global_pca_words.csv"),
        index=False,
        encoding="utf-8"
    )

    rows = []

    for label_safe, sub_model in sub_models_dict.items():
        try:
            R, common = align_submodel_to_global(global_model, sub_model)
        except Exception as e:
            print(f"[WARN] Alignement impossible pour {label_safe}: {e}")
            continue

        for w in common:
            v_sub = sub_model.wv[w].reshape(1, -1)
            v_sub = normalize_rows(v_sub)
            v_sub_aligned = v_sub @ R

            v_global = global_model.wv[w].reshape(1, -1)
            v_global = normalize_rows(v_global)

            xy = pca.transform(v_sub_aligned)[0]

            cos_sim = float(np.dot(v_sub_aligned[0], v_global[0]))
            cos_dist = 1.0 - cos_sim

            rows.append({
                "subcorpus": label_safe,
                "word": w,
                "x": xy[0],
                "y": xy[1],
                "cosine_similarity_to_global": cos_sim,
                "cosine_distance_to_global": cos_dist,
            })

    df_shift = pd.DataFrame(rows)
    df_shift.to_csv(
        os.path.join(shift_dir, "word_positions_by_subcorpus.csv"),
        index=False,
        encoding="utf-8"
    )

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "top_words": top_words,
    }
    with open(
        os.path.join(shift_dir, "global_pca_meta.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Sorties semantic_shift générées.")

#############################################
# Lancement Shiny
#############################################

def run_shiny_and_wait(json_path="subcorpora.json", app_file="shiny_app.py", port="8000"):
    kill_process_on_port(port)

    print(f"Lancement de {app_file}...")

    process = subprocess.Popen(
        [sys.executable, "-m", "shiny", "run", app_file, "--port", port]
    )

    time.sleep(2)

    if process.poll() is not None:
        print(f"[ERREUR] {app_file} ne s'est pas lancé correctement.")
        return None

    url = f"http://127.0.0.1:{port}"
    print(f"Ouverture du navigateur : {url}")
    webbrowser.open(url)

    try:
        while not os.path.exists(json_path):
            print(f"En attente de {json_path}...")
            time.sleep(2)
    finally:
        process.send_signal(signal.SIGINT)
        process.wait()

    return json_path


def run_shiny_viewer(app_file="shiny_vis.py", port="8001", run_name=None):
    kill_process_on_port(port)

    print(f"Lancement de {app_file}...")

    env = os.environ.copy()
    if run_name:
        env["DEFAULT_RUN_NAME"] = run_name

    process = subprocess.Popen(
        [sys.executable, "-m", "shiny", "run", app_file, "--port", str(port)],
        env=env
    )

    time.sleep(2)

    if process.poll() is not None:
        print(f"[ERREUR] {app_file} ne s'est pas lancé correctement.")
        return None

    url = f"http://127.0.0.1:{port}"
    print(f"Ouverture du navigateur : {url}")
    webbrowser.open(url)

    return process

import os
import signal
import subprocess
import sys
import time
import webbrowser


def kill_process_on_port(port):
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        pids = result.stdout.strip().splitlines()

        for pid in pids:
            if pid.strip():
                os.kill(int(pid), signal.SIGKILL)
                print(f"Processus tué sur le port {port} : PID {pid}")

    except Exception as e:
        print(f"[WARN] Impossible de nettoyer le port {port} : {e}")
        
#############################################
# RUN DATA
#############################################
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")


def make_run_name(training_config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    vector_size = training_config.get("vector_size", 300)
    window = training_config.get("window", 2)
    min_count = training_config.get("min_count", 1)
    epochs = training_config.get("epochs", 50)
    sg = training_config.get("sg", 1)
    lowercase = int(bool(training_config.get("lowercase", False)))

    return (
        f"run_{timestamp}"
        f"__sg{sg}"
        f"_dim{vector_size}"
        f"_win{window}"
        f"_mc{min_count}"
        f"_ep{epochs}"
        f"_lc{lowercase}"
    )

def make_run_dirs(run_name):
    run_dir = os.path.join(RUNS_DIR, run_name)

    dirs = {
        "run_dir": run_dir,
        "models": os.path.join(run_dir, "models"),
        "sentences": os.path.join(run_dir, "sentences"),
        "semantic_shift": os.path.join(run_dir, "semantic_shift"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs
        
def save_run_metadata(run_dir, training_config, subcorpora_labels):
    meta = {
        "training_config": training_config,
        "subcorpora": subcorpora_labels,
        "created_at": datetime.now().isoformat()
    }

    path = os.path.join(run_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        
#############################################
# MAIN
#############################################

if __name__ == "__main__":
    ensure_dirs()

    # 1. écran d'accueil
    run_launcher_and_wait(LAUNCHER_JSON, app_file="shiny_start.py", port="7999")
    choice = load_launcher_choice(LAUNCHER_JSON)

    if not choice:
        print("Aucun choix récupéré.")
        sys.exit(1)

    mode = choice.get("mode")

    # 2. ouvrir un run existant
    if mode == "open_existing":
        run_name = choice.get("run_name")
        if not run_name:
            print("Aucun run sélectionné.")
            sys.exit(1)

        print(f"Ouverture du run existant : {run_name}")
        run_shiny_viewer(app_file="shiny_vis.py", port="8001", run_name=run_name)
        sys.exit(0)

    # 3. créer un nouveau run
    if mode == "train_new":
        if os.path.exists(SUBCORPORA_JSON):
            os.remove(SUBCORPORA_JSON)

        run_shiny_and_wait(SUBCORPORA_JSON, app_file="shiny_app.py", port="8000")

        run_name = process_subcorpora(SUBCORPORA_JSON)

        print(f"Traitement terminé pour le run : {run_name}")
        print("Ouverture de l'interface de visualisation...")
        run_shiny_viewer(app_file="shiny_vis.py", port="8001", run_name=run_name)
        sys.exit(0)

    print(f"Mode inconnu : {mode}")
    sys.exit(1)
    
    