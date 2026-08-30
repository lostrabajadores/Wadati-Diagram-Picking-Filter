Run it locally
This page is the companion overview. Jupyter runs on your machine — nothing here executes a local kernel.
01
Generate the files
Writes notebooks/environment.yml and notebooks/wadati_qc.ipynb from the repository.
`python -m app.notebook.build_notebook`

02
Pin conda-forge first
conda-forge at strict priority keeps ObsPy, Panel and the CPU PyTorch build mutually compatible.
`conda config --add channels conda-forge && conda config --set channel_priority strict`

03
Create & activate the environment
Python 3.11 with numpy, scipy, pandas, matplotlib, obspy, panel, pyviz_comms, jupyterlab, CPU torch and seisbench.
`conda env create -f notebooks/environment.yml && conda activate wadati-qc`

04
Open the notebook locally
Run this in your own terminal — this page describes the workflow, it does not start Jupyter for you.
`jupyter lab notebooks/wadati_qc.ipynb`
