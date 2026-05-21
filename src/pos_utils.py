#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:40:52 2026

@author: hugodumoulin
"""

from functools import lru_cache
import spacy


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
