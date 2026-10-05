# Chen 2026 style: one 80/20 split, 3 seeds
# Chen et al. 2026, Nat Comput Sci 6:450-463, doi:10.1038/s43588-026-00972-4

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import ranksums
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC

from lib.config import GF_SETUP, MLP_CONFIG, PUBLISHED_2026, RESULTS
from lib.mlp import mlp

SEEDS = [42, 0, 1]   # 42 = Geneformer's; paper doesn't list its seeds
OUT = RESULTS / "2026_setup"


def run_seed(seed):
    torch.set_num_threads(8)
    y = np.load(GF_SETUP / "genes.npz")["labels"]
    f = np.load(GF_SETUP / "fold0.npz")   # 80/20 split = geneformer_setup's fold 0
    tr, te, ranks, w = f["train"], f["test"], f["ranks"].astype(np.int64), f["occ_eval"][f["test"]]

    score, cfg = mlp(ranks.astype(np.float32) / 2048.0, y, tr, te, seed=seed), MLP_CONFIG["id"]
    # preds = {"MLP": (score, (score > 0).astype(int))}
    preds = {"MLP-r": (score, (score > 0).astype(int))}
    for name, m in [("RF-r", RandomForestClassifier(max_depth=2, random_state=seed)),
                    ("LR-r", LogisticRegression(random_state=0)), ("SVM-r", SVC(random_state=0))]:
        m.fit(ranks[tr], y[tr])
        s = m.decision_function(ranks[te]) if name == "SVM-r" else m.predict_proba(ranks[te])[:, 1]
        preds[name] = (s, m.predict(ranks[te]))
    rows = []
    for name, (s, yhat) in preds.items():
        # rows.append({"seed": seed, "model": name, "config": cfg if name == "MLP" else None,
        rows.append({"seed": seed, "model": name, "config": cfg if name == "MLP-r" else None,
                     "auc_occurrence": roc_auc_score(y[te], s, sample_weight=w),
                     "macro_f1_occurrence": f1_score(y[te], yhat, average="macro", sample_weight=w),
                     "auc_gene": roc_auc_score(y[te], s), "macro_f1_gene": f1_score(y[te], yhat, average="macro")})
        print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"seed{seed}.csv", index=False)


def summarize():
    df = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("seed*.csv"))])
    pub = pd.DataFrame([{"model": k, "seed_index": i, "macro_f1_occurrence": v["macro_f1"][i], "auc_occurrence": v["auc"][i]}
                        for k, v in PUBLISHED_2026.items() for i in range(3)])
    cols = ["macro_f1_occurrence", "auc_occurrence", "macro_f1_gene", "auc_gene"]
    ours = df.groupby("model", sort=False)[cols].agg(["median", "mean", "min", "max"])
    theirs = pub.groupby("model", sort=False)[["macro_f1_occurrence", "auc_occurrence"]].agg(["median", "mean", "min", "max"])
    summary = pd.concat([ours, theirs])
    summary.to_csv(OUT / "summary.csv", float_format="%.4f")
    print(f"{df.seed.nunique()} seeds; per occurrence = geneformer's token-level scoring\n")
    print(summary.round(3).to_string())
    # their test: two-sided Wilcoxon rank sums over seeds
    gf = PUBLISHED_2026["Geneformer GF-10M (published)"]
    # mlp_rows = df[df.model == "MLP"]
    mlp_rows = df[df.model == "MLP-r"]
    print("\nMLP vs published GF-10M, two-sided Wilcoxon rank sums over seeds (different cells; context only):")
    for ours_col, key in [("macro_f1_occurrence", "macro_f1"), ("auc_occurrence", "auc")]:
        r = ranksums(mlp_rows[ours_col], gf[key])
        print(f"  {key:9s} MLP median {mlp_rows[ours_col].median():.3f} vs GF-10M {np.median(gf[key]):.3f}: p = {r.pvalue:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed-index", type=int)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    summarize() if a.summarize else run_seed(SEEDS[a.seed_index])
