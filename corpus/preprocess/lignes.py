#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:46:58 2026

@author: hugodumoulin
"""

import os
import re

def clean_line_breaks(text):
    # Remplace les sauts de ligne non suivis d'une majuscule par un espace
    return re.sub(r'\n\s*(?![A-ZÉÈÀÂÊÎÔÛÇ])', ' ', text)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

            cleaned_text = clean_line_breaks(text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"✔ Traité : {filename}")

# Exemple d'utilisation
input_folder = "/Users/hugodumoulin/Desktop/Python/Word2vec/dev/corpus"
output_folder = "/Users/hugodumoulin/Desktop/Python/Word2vec/dev/corpus+"

process_folder(input_folder, output_folder)