# fixed MLP vs RF/LR/SVM, baseline setup (2023 paper)
# Theodoris et al. 2023, Nature 618:616-624, doi:10.1038/s41586-023-06139-9

import pickle
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from lib.config import (BASELINE_SETUP, DATA, FIGS, FULL_CORPUS, MLP_CONFIG, PUBLISHED_FIG2A, RESULTS, SEED, TF_PICKLE,
                        TOKEN_DICT)
from lib.data import N_CELLS, baseline_folds, contains, rank_matrix
from lib.gf_metrics import cv_metrics, fold_roc, published_geneformer
from lib.mlp import mlp
from lib.plotting import plot_roc

BASELINES = {  # as in geneformer's baseline notebook
    "RF-r": lambda: RandomForestClassifier(max_depth=2, random_state=0),
    "LR-r": lambda: LogisticRegression(random_state=0),
    "SVM-r": lambda: SVC(random_state=0),
}


def build_setup():
    # 122/122 balanced genes, fixed 10k cells
    token_dict = pickle.load(open(TOKEN_DICT, "rb"))
    tfs = pickle.load(open(TF_PICKLE, "rb"))
    t1 = [g for g in tfs["Dosage-sensitive TFs"] if g in token_dict]
    t2 = [g for g in tfs["Dosage-insensitive TFs"] if g in token_dict]
    n = min(len(t1), len(t2))
    random.seed(0)
    t1 = random.sample(t1, n)
    random.seed(0)
    t2 = random.sample(t2, n)
    genes = t1 + t2
    tokens = np.array([token_dict[g] for g in genes])
    labels = np.array([0] * n + [1] * n)          # 0 = sensitive, 1 = insensitive

    ds = load_from_disk(str(FULL_CORPUS)).shuffle(seed=42, keep_in_memory=True)
    ds = ds.select(range(N_CELLS * 5)).filter(contains(tokens), num_proc=16).select(range(N_CELLS))
    ranks = rank_matrix(ds, tokens)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BASELINE_SETUP, ranks=ranks, labels=labels, tokens=tokens, genes=np.array(genes))
    print(f"built setup: {len(genes)} genes ({n}/{n}), {ranks.shape[1]} cells")


def main():
    torch.set_num_threads(8)
    if not BASELINE_SETUP.exists():
        build_setup()
    d = np.load(BASELINE_SETUP)
    ranks, y = d["ranks"].astype(np.int64), d["labels"]
    X = ranks.astype(np.float32) / 2048.0

    rows, per = [], {}
    for k, (tr, te) in enumerate(baseline_folds(y)):
        preds = {}
        # preds["MLP"], cfg = mlp(X, y, tr, te, seed=SEED), MLP_CONFIG["id"]
        preds["MLP-r"], cfg = mlp(X, y, tr, te, seed=SEED), MLP_CONFIG["id"]
        for name, make in BASELINES.items():
            m = make().fit(ranks[tr], y[tr])
            preds[name] = m.decision_function(ranks[te]) if name == "SVM-r" else m.predict_proba(ranks[te])[:, 1]
        for name, s in preds.items():
            auc, tpr, wt = fold_roc(y[te], s)
            for lst, v in zip(per.setdefault(name, ([], [], [])), (auc, tpr, wt)):
                lst.append(v)
            # rows.append({"model": name, "fold": k, "config": cfg if name == "MLP" else None, "auc_gene": auc,
            rows.append({"model": name, "fold": k, "config": cfg if name == "MLP-r" else None, "auc_gene": auc,
                         "tpr_wt": wt})
        # print(f"fold {k}: MLP config {cfg}, AUC {per['MLP'][0][-1]:.3f}", flush=True)
        print(f"fold {k}: MLP config {cfg}, AUC {per['MLP-r'][0][-1]:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "baseline_setup_per_fold.csv", index=False)

    # curves = [(name, *cv_metrics(*per[name])) for name in ["MLP", "SVM-r", "RF-r", "LR-r"]]
    curves = [(name, *cv_metrics(*per[name])) for name in ["MLP-r", "SVM-r", "RF-r", "LR-r"]]
    # a_mlp = np.array(per["MLP"][0])
    a_mlp = np.array(per["MLP-r"][0])
    print("model   weighted AUC ± weighted SD (per gene)   published Fig. 2a")
    for name, m, sd, _ in curves:
        # pub = "" if name == "MLP" else "%.2f ± %.2f" % PUBLISHED_FIG2A[name]
        pub = "" if name == "MLP-r" else "%.2f ± %.2f" % PUBLISHED_FIG2A[name]
        print(f"{name:6s}  {m:.3f} ± {sd:.3f}                        {pub}")
    for name in ["SVM-r", "RF-r", "LR-r"]:
        a = np.array(per[name][0])
        print(f"MLP vs {name}: {np.mean(a_mlp - a):+.3f} (wins {(a_mlp > a).sum()}/5)")

    gf = published_geneformer()
    plot_roc([("Geneformer (published)", gf["weighted_auc"], gf["weighted_sd"], gf["mean_tpr"])] + curves,
             # "MLP in Geneformer's baseline setup", FIGS / "baseline_setup",
             "MLP-r in Geneformer's baseline setup", FIGS / "baseline_setup",
             # "MLP + baselines: 122/122 genes, per gene\nGeneformer: 490 genes, per occurrence")
             "MLP-r + baselines: 122/122 genes, per gene\nGeneformer: 490 genes, per occurrence")


if __name__ == "__main__":
    main()
