import reflex as rx
from typing import TypedDict


class Step(TypedDict):
    index: str
    title: str
    detail: str
    command: str


class SetupState(rx.State):
    steps: list[Step] = [
        Step(
            index="01",
            title="Generate the files",
            detail="Writes notebooks/environment.yml and notebooks/wadati_qc.ipynb from the repository.",
            command="python -m app.notebook.build_notebook",
        ),
        Step(
            index="02",
            title="Pin conda-forge first",
            detail="conda-forge at strict priority keeps ObsPy, Panel and the CPU PyTorch build mutually compatible.",
            command="conda config --add channels conda-forge && conda config --set channel_priority strict",
        ),
        Step(
            index="03",
            title="Create & activate the environment",
            detail="Python 3.11 with numpy, scipy, pandas, matplotlib, obspy, panel, pyviz_comms, jupyterlab, CPU torch and seisbench.",
            command="conda env create -f notebooks/environment.yml && conda activate wadati-qc",
        ),
        Step(
            index="04",
            title="Open the notebook locally",
            detail="Run this in your own terminal — this page describes the workflow, it does not start Jupyter for you.",
            command="jupyter lab notebooks/wadati_qc.ipynb",
        ),
    ]

    copied: str = ""

    @rx.event
    def copy(self, command: str):
        self.copied = command
        yield rx.set_clipboard(command)
        yield rx.toast("Command copied to clipboard", duration=2500)
