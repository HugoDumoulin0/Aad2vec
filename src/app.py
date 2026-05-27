#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:41:43 2026

@author: hugodumoulin
"""

from shiny import App
import webbrowser
import threading

from ui import app_ui
from server import server


app = App(app_ui, server)

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(port=8000)