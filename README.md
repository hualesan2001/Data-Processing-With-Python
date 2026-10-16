# FMA Music Dataset Analysis (DPiP Project)

## About

This project analyses the **Free Music Archive (FMA)** — a public dataset of ~106 574 Creative-Commons audio tracks released by Defferrard et al. (2017) — using its metadata (artist, album, title, top-level genre, release date) and 518 pre-computed `librosa` audio features. The analysis is delivered as a Jupyter notebook (`report.ipynb`) that runs three pipelines end-to-end: per-decade PCA + k-means clustering, lexical exploration of song titles, and a genre-pair PCA centroid-distance study over 5-year periods. The goal is to look for patterns and drift in audio-feature space and song-title vocabulary across genres and time.

## Install

Python 3.10+ is recommended. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `fma_analysis` package (defined in `pyproject.toml`) in editable mode so the notebook and CLI scripts can `import fma_analysis.*`. JupyterLab and the Python kernel are included in `requirements.txt`, so no separate install step is needed.

## Run

Open the notebook:

```bash
jupyter lab report.ipynb
```

Run all cells top-to-bottom. **The first non-intro cell (`download-data`) will automatically fetch the FMA metadata + `fma_small` audio archive into `./data/`** the first time you run it; subsequent runs detect the existing folders and skip the download. Allow a few GB of disk space and a few minutes for the first fetch.

> **Note on download issues**: the FMA dataset is hosted on `os.unil.cloud.switch.ch`. Some ISPs and "safe-browsing" / parental-control services flag this host as suspicious and silently block the request. If the download cell hangs or errors out, **temporarily disable your ISP's safe-browsing filter** (or switch ISPs / use a different network) and retry. The cell is idempotent, so just re-running it resumes from where it left off.
