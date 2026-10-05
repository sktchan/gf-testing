# ROC plot and curve export (GF style for comparison i guess)

from lib.config import FIGS, RESULTS
from lib.gf_metrics import MEAN_FPR

STYLE = {  # colors as in the paper's Fig. 2a
    # "Geneformer (published)": ("red", "-"), "MLP": ("darkorange", "-"),
    "Geneformer (published)": ("red", "-"), "MLP-r": ("darkorange", "-"),
    "SVM-r": ("purple", "-"), "RF-r": ("blue", "-"), "LR-r": ("green", "-"),
    # "Detection count": ("gray", "--"),
    "Count": ("gray", "--"),
}


def save_curves(curves, path):
    # ROC curves as long csv for plots.jl
    import pandas as pd
    pd.DataFrame([{"model": n, "auc": m, "sd": sd, "fpr": f, "tpr": t}
                  for n, m, sd, tpr in curves for f, t in zip(MEAN_FPR, tpr)]).to_csv(path, index=False)


def plot_roc(curves, title, path, subtitle):
    # curves: list of (name, auc, sd, mean_tpr)
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_curves(curves, RESULTS / f"{path.name}_roc.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, m, sd, tpr in curves:
        color, ls = STYLE.get(name, ("black", "-"))
        ax.plot(MEAN_FPR, tpr, color=color, ls=ls, lw=2, label=f"{name}: {m:.2f} ± {sd:.2f}")
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title(title, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title=f"AUC ± SD by 5-fold cross-validation (fold-weighted, as Geneformer)\n{subtitle}",
              title_fontsize=9, fontsize=9,
              frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
