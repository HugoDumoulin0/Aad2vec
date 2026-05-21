#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:24:41 2026

@author: hugodumoulin
"""

import json
import os
import subprocess
import sys


LAUNCHER_JSON = "launcher_choice.json"
SUBCORPORA_JSON = "subcorpora.json"

PYTHON = sys.executable


def run_step(label, cmd):
    print("")
    print("=" * 60)
    print(label)
    print("=" * 60)
    print("Commande :", " ".join(cmd))
    print("Quand l'étape est terminée, arrête l'app avec Ctrl+C dans ce terminal.")
    print("")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print(f"{label} arrêté.")
    except subprocess.CalledProcessError as exc:
        print(f"Erreur pendant : {label}")
        raise exc


def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    run_step(
        "Étape 1 - Choix du mode",
        [PYTHON, "launcher.py"],
    )

    choice = load_json(LAUNCHER_JSON)

    if not choice:
        print(f"{LAUNCHER_JSON} introuvable. Pipeline arrêté.")
        return

    mode = choice.get("mode")

    if mode == "train_new":
        run_step(
            "Étape 2 - Configuration du nouveau run",
            [PYTHON, "configure_runs.py"],
        )

        if not os.path.exists(SUBCORPORA_JSON):
            print(f"{SUBCORPORA_JSON} introuvable. Pipeline arrêté.")
            return

        run_step(
            "Étape 3 - Entraînement",
            [PYTHON, "train.py"],
        )

    elif mode == "open_existing":
        run_name = choice.get("run_name")
        if run_name:
            os.environ["DEFAULT_RUN_NAME"] = run_name

    else:
        print(f"Mode inconnu : {mode}")
        return

    run_step(
        "Étape finale - Visualisation",
        [PYTHON, "app.py"],
    )


if __name__ == "__main__":
    main()
