#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

from shiny import App, ui, reactive, render

from loaders import available_runs


EXPORT_JSON = "launcher_choice.json"


app_ui = ui.page_fluid(
    ui.h1("Aad2Vec"),
    ui.h2("Choix du mode de travail"),

    ui.input_radio_buttons(
        "entry_mode",
        "Que faire ?",
        choices={
            "open_existing": "Ouvrir un run existant",
            "train_new": "Créer / réentraîner un nouveau run",
        },
        selected="open_existing",
    ),

    ui.output_ui("run_selector_ui"),

    ui.hr(),

    ui.input_action_button("confirm", "Continuer"),

    ui.hr(),

    ui.h4("Statut"),
    ui.output_text("status_text"),
)


def server(input, output, session):
    status = reactive.Value("En attente...")

    @output
    @render.ui
    def run_selector_ui():
        mode = input.entry_mode()

        if mode != "open_existing":
            return ui.p("Un nouveau run sera créé après segmentation et choix des hyperparamètres.")

        runs = available_runs()

        if not runs:
            return ui.p("Aucun run existant trouvé dans ../outputs/runs")

        return ui.input_select(
            "selected_run",
            "Choisir un run existant",
            choices=runs,
            selected=runs[0],
        )

    @output
    @render.text
    def status_text():
        return status.get()

    @reactive.Effect
    @reactive.event(input.confirm)
    def _confirm():
        mode = input.entry_mode()
        payload = {"mode": mode}

        if mode == "open_existing":
            runs = available_runs()
            selected_run = input.selected_run() if hasattr(input, "selected_run") else None

            if not runs:
                status.set("Aucun run existant disponible.")
                return

            if not selected_run:
                status.set("Aucun run sélectionné.")
                return

            payload["run_name"] = selected_run

        with open(EXPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        status.set(f"Choix sauvegardé dans {EXPORT_JSON}")


app = App(app_ui, server)


if __name__ == "__main__":
    app.run(port=8001)
