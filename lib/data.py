# rank-value features, occurrences, baseline-setup folds

import numpy as np
from sklearn.model_selection import StratifiedKFold

NUM_PROC = 16
N_CELLS = 10_000


def contains(targets):
    s = set(int(t) for t in targets)
    return lambda ex: not s.isdisjoint(ex["input_ids"])


def rank_matrix(ds, tokens):
    # genes x cells: 2048 - position, 0 if absent
    tokens = np.asarray(tokens)

    def get_ranks(example):
        d = dict(zip(example["input_ids"], [2048 - i for i in range(example["length"])]))
        example["target_vector"] = [d.get(t, 0) for t in tokens]
        return example

    return np.asarray(ds.map(get_ranks, num_proc=NUM_PROC)["target_vector"], dtype=np.int16).T


def occurrences(ds, tokens):
    # cells containing each token = tokens geneformer scores
    idx = {int(t): i for i, t in enumerate(tokens)}
    n = np.zeros(len(tokens), dtype=np.int64)
    for ids in ds["input_ids"]:
        for t in idx.keys() & set(ids):
            n[idx[t]] += 1
    return n


def baseline_folds(y):
    # geneformer baseline notebook's folds over the 244 genes
    return list(StratifiedKFold(n_splits=5, random_state=0, shuffle=True).split(np.zeros(len(y)), y))
