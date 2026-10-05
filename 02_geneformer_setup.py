# 02: fixed MLP and baselines, scored exactly like Geneformer (2023 paper)
# Theodoris et al. 2023, Nature 618:616-624, doi:10.1038/s41586-023-06139-9
# via huggingface.co/ctheodoris/Geneformer examples/gene_classification.ipynb

import pickle

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from lib.config import CELLS_50K, FIGS, GF_SETUP, MLP_CONFIG, RESULTS, SEED, TF_PICKLE, TOKEN_DICT
from lib.data import N_CELLS, contains, occurrences, rank_matrix
from lib.gf_metrics import cv_metrics, fold_roc, published_geneformer, save_published
from lib.mlp import mlp
from lib.plotting import plot_roc

BASELINES = {
    "RF-r": lambda: RandomForestClassifier(max_depth=2, random_state=0),
    "LR-r": lambda: LogisticRegression(random_state=0),
    "SVM-r": lambda: SVC(random_state=0),
}


def build_setup():
    # Geneformer Classifier @9e9cca9: 490 genes, per-fold 10k cells
    token_dict = pickle.load(open(TOKEN_DICT, "rb"))
    tfs = pickle.load(open(TF_PICKLE, "rb"))
    gene_class_dict = {k: set(token_dict.get(g) for g in v) for k, v in tfs.items()}
    class_id = {k: i for i, k in enumerate(gene_class_dict)}          # sensitive 0, insensitive 1
    tokens = np.array([t for v in gene_class_dict.values() for t in v])
    labels = np.array([class_id[k] for k, v in gene_class_dict.items() for _ in v])

    data = load_from_disk(str(CELLS_50K)).filter(contains(tokens), num_proc=16)
    assert len(data) == 33558, f"{len(data)} cells, published run had 33558"
    data = data.shuffle(seed=42)

    GF_SETUP.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for k, (tr, ev) in enumerate(skf.split(tokens, labels)):
        train_cells = data.filter(contains(tokens[tr]), num_proc=16).shuffle(seed=42)
        train_cells = train_cells.select(range(min(N_CELLS, len(train_cells))))
        eval_cells = data.filter(contains(tokens[ev]), num_proc=16).shuffle(seed=42)
        eval_cells = eval_cells.select(range(min(N_CELLS, len(eval_cells))))
        # features from training cells; scores weighted by eval occurrences
        np.savez_compressed(GF_SETUP / f"fold{k}.npz", train=tr, test=ev,
                            ranks=rank_matrix(train_cells, tokens), occ_eval=occurrences(eval_cells, tokens))
        print(f"fold {k}: {len(tr)} train / {len(ev)} eval genes", flush=True)
    np.savez_compressed(GF_SETUP / "genes.npz", tokens=tokens, labels=labels)


def main():
    torch.set_num_threads(8)
    if not (GF_SETUP / "genes.npz").exists():
        build_setup()
    y = np.load(GF_SETUP / "genes.npz")["labels"]

    rows, per = [], {}
    for k in range(5):
        f = np.load(GF_SETUP / f"fold{k}.npz")
        tr, te, ranks, w = f["train"], f["test"], f["ranks"].astype(np.int64), f["occ_eval"][f["test"]]
        preds = {}
        # preds["MLP"], cfg = mlp(ranks.astype(np.float32) / 2048.0, y, tr, te, seed=SEED), MLP_CONFIG["id"]
        preds["MLP-r"], cfg = mlp(ranks.astype(np.float32) / 2048.0, y, tr, te, seed=SEED), MLP_CONFIG["id"]
        for name, make in BASELINES.items():
            m = make().fit(ranks[tr], y[tr])
            preds[name] = m.decision_function(ranks[te]) if name == "SVM-r" else m.predict_proba(ranks[te])[:, 1]
        # preds["Detection count"] = -(ranks[te] > 0).sum(1).astype(float)   # untrained detection baseline
        preds["Count"] = -(ranks[te] > 0).sum(1).astype(float)   # untrained detection baseline: cells detecting the gene
        for name, s in preds.items():
            auc_occ, tpr, wt = fold_roc(y[te], s, weight=w)
            auc_gene, tpr_g, wt_g = fold_roc(y[te], s)
            p = per.setdefault(name, {"occ": ([], [], []), "gene": ([], [], [])})
            for key, vals in (("occ", (auc_occ, tpr, wt)), ("gene", (auc_gene, tpr_g, wt_g))):
                for lst, v in zip(p[key], vals):
                    lst.append(v)
            rows.append({"model": name, "fold": k, "auc_occurrence": auc_occ, "tpr_wt_occurrence": wt,
                         # "auc_gene": auc_gene, "tpr_wt_gene": wt_g, "config": cfg if name == "MLP" else None})
                         "auc_gene": auc_gene, "tpr_wt_gene": wt_g, "config": cfg if name == "MLP-r" else None})
        print(f"fold {k}: MLP config {cfg}, per-occurrence AUC {rows[-5]['auc_occurrence']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "02_geneformer_setup_per_fold.csv", index=False)
    gf = published_geneformer()
    save_published()
    print(f"\ngeneformer (published): {gf['weighted_auc']:.3f} ± {gf['weighted_sd']:.3f} per occurrence "
          f"(weighted, as reported); unweighted fold mean {gf['folds'].mean():.3f}")
    summary, curves = [], [("Geneformer (published)", gf["weighted_auc"], gf["weighted_sd"], gf["mean_tpr"])]
    # for name in ["MLP", "SVM-r", "RF-r", "LR-r", "Detection count"]:
    # for name in ["MLP", "SVM-r", "RF-r", "LR-r", "Count"]:
    for name in ["MLP-r", "SVM-r", "RF-r", "LR-r", "Count"]:
        occ, gene = cv_metrics(*per[name]["occ"]), cv_metrics(*per[name]["gene"])
        a = np.array(per[name]["occ"][0])
        summary.append({"model": name, **{f"fold{i}": v for i, v in enumerate(a)},
                        "auc_occurrence": occ[0], "sd_occurrence": occ[1],
                        "auc_gene": gene[0], "sd_gene": gene[1],
                        "minus_geneformer": occ[0] - gf["weighted_auc"],
                        "wins_vs_geneformer": int((a > gf["folds"]).sum()),
                        "p_wilcoxon_vs_geneformer": wilcoxon(a, gf["folds"]).pvalue})
        curves.append((name, *occ))
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "02_geneformer_setup_summary.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False, float_format="%.3f"))

    plot_roc(curves, "Dosage sensitive vs. insensitive TFs", FIGS / "fig2_geneformer_setup",
             "Geneformer's evaluation: 490 genes,\nits folds and cells, per gene occurrence")


if __name__ == "__main__":
    main()
