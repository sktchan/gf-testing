# loads config/*.toml into constants

from pathlib import Path

import tomli

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    with open(ROOT / "config" / name, "rb") as f:
        return tomli.load(f)


_paths = _load("paths.toml")
FULL_CORPUS = ROOT / _paths["full_corpus"]
TOKEN_DICT = ROOT / _paths["token_dict"]
TF_PICKLE = ROOT / _paths["tf_pickle"]
CELLS_50K = ROOT / _paths["cells_50k"]
GF_NOTEBOOK = ROOT / _paths["gf_notebook"]
DATA = ROOT / _paths["data"]
BASELINE_SETUP = ROOT / _paths["baseline_setup"]
GF_SETUP = ROOT / _paths["gf_setup"]
RESULTS = ROOT / _paths["results"]
FIGS = ROOT / _paths["figures"]

_mlp = _load("mlp.toml")
SEED = _mlp["seed"]
MLP_CONFIG = dict(_mlp["mlp"], family="dense")

_published = _load("published.toml")
PUBLISHED_FIG2A = {k: tuple(v) for k, v in _published["fig2a_2023"].items()}
PUBLISHED_2026 = _published["chen_2026"]
