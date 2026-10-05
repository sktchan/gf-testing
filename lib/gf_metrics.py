# geneformer's metrics/results

import json
import re

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from lib.config import GF_NOTEBOOK, RESULTS

MEAN_FPR = np.linspace(0, 1, 100)                   # geneformer's ROC grid


def fold_roc(y, score, weight=None):
    # AUC, interpolated ROC, len(tpr) fold weight; weight = occurrences
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    interp = np.interp(MEAN_FPR, fpr, tpr)
    interp[0] = 0.0
    return roc_auc_score(y, score, sample_weight=weight), interp, len(tpr)


def cv_metrics(aucs, tprs, tpr_wts):
    # geneformer's get_cross_valid_metrics: len(tpr)-weighted AUC ± SD
    wts = np.asarray(tpr_wts, dtype=float) / np.sum(tpr_wts)
    mean_tpr = np.sum(np.asarray(tprs) * wts[:, None], axis=0)
    mean_tpr[-1] = 1.0
    roc_auc = float(np.sum(np.asarray(aucs) * wts))
    roc_auc_sd = float(np.sqrt(np.average((np.asarray(aucs) - roc_auc) ** 2, weights=wts)))
    return roc_auc, roc_auc_sd, mean_tpr


def published_geneformer():
    # published run's fold AUCs and ROC, parsed from notebook
    nb = json.load(open(GF_NOTEBOOK))
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            t = "".join(o.get("data", {}).get("text/plain", ""))
            if "mean_tpr" in t and "all_roc_auc" in t:
                arr = lambda k: np.array([float(x) for x in
                                          re.search(rf"'{k}': array\(\[([^\]]*)\]\)", t).group(1).split(",")])
                folds = [float(x) for x in re.search(r"'all_roc_auc': \[([^\]]*)\]", t).group(1).split(",")]
                weighted = float(re.search(r"'roc_auc': ([0-9.]+)", t).group(1))
                weighted_sd = float(re.search(r"'roc_auc_sd': ([0-9.]+)", t).group(1))
                return {"folds": np.array(folds), "mean_fpr": arr("mean_fpr"), "mean_tpr": arr("mean_tpr"),
                        "weighted_auc": weighted, "weighted_sd": weighted_sd}
    raise RuntimeError("published geneformer metrics not found in notebook")


def save_published(path=RESULTS / "geneformer_published.csv"):
    # published geneformer fold AUCs for plots.jl
    import pandas as pd
    gf = published_geneformer()
    pd.DataFrame({"fold": range(len(gf["folds"])), "auc": gf["folds"],
                  "weighted_auc": gf["weighted_auc"], "weighted_sd": gf["weighted_sd"]}).to_csv(path, index=False)
