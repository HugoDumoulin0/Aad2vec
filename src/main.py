#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:49:45 2026

@author: hugodumoulin
"""

import os
import json
import time
import signal
import subprocess
import sys
import webbrowser
import re
import unicodedata
import urllib.request


from datetime import datetime
from collections import defaultdict

import spacy
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes


CORPUS_PATH = "../corpus"
OUTPUT_DIR = "../outputs"
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")

LAUNCHER_JSON = "launcher_choice.json"
SUBCORPORA_JSON = "subcorpora.json"
SPACY_MAX_CHARS_PER_CHUNK = 200_000

def kill_process_on_port(port):
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        pids = result.stdout.strip().splitlines()

        for pid in pids:
            if pid.strip():
                os.kill(int(pid), signal.SIGKILL)
                print(f"Processus tué sur le port {port} : PID {pid}")

    except Exception as exc:
        print(f"[WARN] Impossible de nettoyer le port {port} : {exc}")

def wait_for_server(url, timeout=30):
    start = time.time()

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(0.5)

    return False


def run_shiny_and_wait(json_path, app_file, port):
    kill_process_on_port(port)

    if os.path.exists(json_path):
        os.remove(json_path)

    print(f"Lancement de {app_file}...")

    process = subprocess.Popen(
        [sys.executable, "-m", "shiny", "run", app_file, "--port", str(port)]
    )

    url = f"http://127.0.0.1:{port}"
    
    if not wait_for_server(url, timeout=30):
        if process.poll() is not None:
            print(f"[ERREUR] {app_file} s'est arrêté avant d'être prêt.")
        else:
            print(f"[ERREUR] {app_file} ne répond pas sur {url}.")
        return None
    
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


def run_shiny_viewer(app_file="app.py", port="8001", run_name=None):
    kill_process_on_port(port)

    print(f"Lancement de {app_file}...")

    env = os.environ.copy()
    if run_name:
        env["DEFAULT_RUN_NAME"] = run_name

    process = subprocess.Popen(
        [sys.executable, "-m", "shiny", "run", app_file, "--port", str(port)],
        env=env,
    )

    url = f"http://127.0.0.1:{port}"
    
    if not wait_for_server(url, timeout=30):
        if process.poll() is not None:
            print(f"[ERREUR] {app_file} s'est arrêté avant d'être prêt.")
        else:
            print(f"[ERREUR] {app_file} ne répond pas sur {url}.")
        return None
    
    print(f"Ouverture du navigateur : {url}")
    webbrowser.open(url)
    
    return process


def load_launcher_choice(path_json=LAUNCHER_JSON):
    if not os.path.exists(path_json):
        return None

    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)


def safe_label(label):
    label = unicodedata.normalize("NFKD", str(label))
    label = label.encode("ascii", "ignore").decode("ascii")
    label = label.lower().strip()
    label = re.sub(r"[^\w\s-]", "_", label)
    label = re.sub(r"[\s/\\]+", "_", label)
    label = re.sub(r"_+", "_", label)
    label = label.strip("_")

    if not label:
        label = "groupe"

    return label


def load_subcorpora_and_config(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "subcorpora" not in data:
        return data, {}

    subcorpora_groups = data.get("subcorpora", {})
    training_config = data.get("training_config", {})

    return subcorpora_groups, training_config


def build_subcorpora_from_files_dict(groups, base_path=CORPUS_PATH):
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


def iter_spacy_chunks(text, max_chars=SPACY_MAX_CHARS_PER_CHUNK):
    text = str(text)
    start = 0
    n_chars = len(text)

    while start < n_chars:
        end = min(start + max_chars, n_chars)

        if end < n_chars:
            split_at = -1
            for sep in ("\n\n", "\n", ". ", "! ", "? ", "… ", "; "):
                idx = text.rfind(sep, start, end)
                if idx > start + max_chars // 2:
                    split_at = idx + len(sep)
                    break

            if split_at == -1:
                idx = text.rfind(" ", start, end)
                if idx > start:
                    split_at = idx + 1

            if split_at != -1:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        start = end


def tokenize(corpus_liste, lowercase=True, token_representation="word"):
    nlp = spacy.load("fr_core_news_sm")
    nlp.max_length = max(nlp.max_length, SPACY_MAX_CHARS_PER_CHUNK + 1)
    corpus_sentences = []

    for texte in corpus_liste:
        for chunk in iter_spacy_chunks(texte):
            doc = nlp(chunk, disable=["ner"])

            for sent in doc.sents:
                word_tokens = []
                lemma_tokens = []

                for token in sent:
                    if token.is_space or token.is_punct:
                        continue

                    word = str(token.text).strip()
                    lemma = str(token.lemma_).strip()

                    if not word:
                        continue

                    if not lemma or lemma == "-PRON-":
                        lemma = word

                    if lowercase:
                        word = word.lower()
                        lemma = lemma.lower()

                    word_tokens.append(word)
                    lemma_tokens.append(lemma)

                if not word_tokens:
                    continue

                train_tokens = lemma_tokens if token_representation == "lemma" else word_tokens

                corpus_sentences.append(
                    {
                        "tokens": train_tokens,
                        "word": word_tokens,
                        "lemma": lemma_tokens,
                        "surface_text": " ".join(word_tokens),
                    }
                )

    return corpus_sentences


def save_sentences(sentences, label_safe, sentences_dir):
    path = os.path.join(sentences_dir, f"{label_safe}_sentences.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print(f"Phrases sauvegardées : {path} ({len(sentences)} phrases)")


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
                compute_loss=True,
            )
            loss = w2v_model.get_latest_training_loss()
            elapsed = time.time() - start_time

            print(
                f"{os.path.basename(model_path)} | Epoch {epoch + 1}/{epochs} | "
                f"Loss: {loss:.2f} | Temps: {elapsed:.2f}s"
            )

        w2v_model.save(model_path)
    else:
        print(f"Chargement du modèle existant : {model_path}")
        w2v_model = Word2Vec.load(model_path)

    return w2v_model


def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def align_submodel_to_global(global_model, sub_model, anchor_topn=5000, min_anchor=200):
    global_vocab = set(global_model.wv.key_to_index.keys())
    sub_vocab = set(sub_model.wv.key_to_index.keys())

    common = list(global_vocab & sub_vocab)
    common = sorted(
        common,
        key=lambda word: global_model.wv.get_vecattr(word, "count"),
        reverse=True,
    )[:anchor_topn]

    if len(common) < min_anchor:
        raise ValueError(
            f"Pas assez de mots ancres pour l'alignement ({len(common)} < {min_anchor})"
        )

    X_sub = np.vstack([sub_model.wv[word] for word in common])
    X_glob = np.vstack([global_model.wv[word] for word in common])

    X_sub = normalize_rows(X_sub)
    X_glob = normalize_rows(X_glob)

    R, _ = orthogonal_procrustes(X_sub, X_glob)

    return R, common


def build_semantic_shift_outputs(global_model, sub_models_dict, shift_dir, top_words=1000):
    os.makedirs(shift_dir, exist_ok=True)

    global_words = list(global_model.wv.index_to_key)[:top_words]
    X_global = np.vstack([global_model.wv[word] for word in global_words])
    X_global = normalize_rows(X_global)

    pca = PCA(n_components=2)
    X_global_2d = pca.fit_transform(X_global)

    df_global = pd.DataFrame(
        {
            "word": global_words,
            "x": X_global_2d[:, 0],
            "y": X_global_2d[:, 1],
        }
    )

    df_global.to_csv(
        os.path.join(shift_dir, "global_pca_words.csv"),
        index=False,
        encoding="utf-8",
    )

    rows = []

    for label_safe, sub_model in sub_models_dict.items():
        try:
            R, common = align_submodel_to_global(global_model, sub_model)
        except Exception as exc:
            print(f"[WARN] Alignement impossible pour {label_safe}: {exc}")
            continue

        for word in common:
            v_sub = sub_model.wv[word].reshape(1, -1)
            v_sub = normalize_rows(v_sub)
            v_sub_aligned = v_sub @ R

            v_global = global_model.wv[word].reshape(1, -1)
            v_global = normalize_rows(v_global)

            xy = pca.transform(v_sub_aligned)[0]

            cos_sim = float(np.dot(v_sub_aligned[0], v_global[0]))
            cos_dist = 1.0 - cos_sim

            rows.append(
                {
                    "subcorpus": label_safe,
                    "word": word,
                    "x": xy[0],
                    "y": xy[1],
                    "cosine_similarity_to_global": cos_sim,
                    "cosine_distance_to_global": cos_dist,
                }
            )

    df_shift = pd.DataFrame(rows)

    df_shift.to_csv(
        os.path.join(shift_dir, "word_positions_by_subcorpus.csv"),
        index=False,
        encoding="utf-8",
    )

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "top_words": top_words,
    }

    with open(
        os.path.join(shift_dir, "global_pca_meta.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Sorties semantic_shift générées.")


def make_run_name(training_config):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    vector_size = int(training_config.get("vector_size", 300))
    window = int(training_config.get("window", 2))
    min_count = int(training_config.get("min_count", 1))
    epochs = int(training_config.get("epochs", 50))
    sg = int(training_config.get("sg", 1))
    negative = int(training_config.get("negative", 5))
    sample = training_config.get("sample", 0.001)
    lowercase = int(bool(training_config.get("lowercase", True)))
    token_representation = str(training_config.get("token_representation", "word"))

    token_tag = "lemma" if token_representation == "lemma" else "word"
    sample_tag = str(sample).replace(".", "p")

    return (
        f"run_{now}"
        f"__{token_tag}"
        f"__lc{lowercase}"
        f"__sg{sg}"
        f"__dim{vector_size}"
        f"__win{window}"
        f"__mc{min_count}"
        f"__ep{epochs}"
        f"__neg{negative}"
        f"__s{sample_tag}"
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
        "created_at": datetime.now().isoformat(),
    }

    path = os.path.join(run_dir, "metadata.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def process_subcorpora(path_json):
    ensure_dirs()

    groups, training_config = load_subcorpora_and_config(path_json)
    subcorpora = build_subcorpora_from_files_dict(groups)

    run_name = make_run_name(training_config)
    dirs = make_run_dirs(run_name)

    sub_models_dict = {}

    lowercase = bool(training_config.get("lowercase", False))
    token_representation = str(training_config.get("token_representation", "word"))

    for label, textes in subcorpora.items():
        print(f"\n=== Traitement du corpus : {label} ===")

        sentences = tokenize(
            textes,
            lowercase=lowercase,
            token_representation=token_representation,
        )

        if not sentences:
            print("[WARN] Corpus vide après tokenisation.")
            continue

        train_sentences = [sentence["tokens"] for sentence in sentences]
        label_safe = safe_label(label)

        save_sentences(sentences, label_safe, dirs["sentences"])

        model_path = os.path.join(dirs["models"], f"w2v_{label_safe}.model")
        model = train(train_sentences, model_path, training_config)

        sub_models_dict[label_safe] = model

    all_texts = []
    for textes in subcorpora.values():
        all_texts.extend(textes)

    global_sentences = tokenize(
        all_texts,
        lowercase=lowercase,
        token_representation=token_representation,
    )

    global_train_sentences = [sentence["tokens"] for sentence in global_sentences]
    global_model_path = os.path.join(dirs["models"], "w2v_global.model")
    global_model = train(global_train_sentences, global_model_path, training_config)

    build_semantic_shift_outputs(
        global_model=global_model,
        sub_models_dict=sub_models_dict,
        shift_dir=dirs["semantic_shift"],
        top_words=1000,
    )

    save_sentences(global_sentences, "global", dirs["sentences"])

    save_run_metadata(
        run_dir=dirs["run_dir"],
        training_config=training_config,
        subcorpora_labels=list(sub_models_dict.keys()),
    )

    return run_name


if __name__ == "__main__":
    ensure_dirs()

    run_shiny_and_wait(
        json_path=LAUNCHER_JSON,
        app_file="launcher.py",
        port="7999",
    )

    choice = load_launcher_choice(LAUNCHER_JSON)

    if not choice:
        print("Aucun choix récupéré.")
        sys.exit(1)

    mode = choice.get("mode")

    if mode == "open_existing":
        run_name = choice.get("run_name")

        if not run_name:
            print("Aucun run sélectionné.")
            sys.exit(1)

        print(f"Ouverture du run existant : {run_name}")

        run_shiny_viewer(
            app_file="app.py",
            port="8001",
            run_name=run_name,
        )

        sys.exit(0)

    if mode == "train_new":
        run_shiny_and_wait(
            json_path=SUBCORPORA_JSON,
            app_file="configure_runs.py",
            port="8000",
        )

        run_name = process_subcorpora(SUBCORPORA_JSON)

        print(f"Traitement terminé pour le run : {run_name}")
        print("Ouverture de l'interface de visualisation...")

        run_shiny_viewer(
            app_file="app.py",
            port="8001",
            run_name=run_name,
        )

        sys.exit(0)

    print(f"Mode inconnu : {mode}")
    sys.exit(1)
