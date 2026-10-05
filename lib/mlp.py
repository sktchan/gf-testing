# fixed MLP 

import torch
import torch.nn as nn

from lib.config import MLP_CONFIG, SEED


def build_mlp(c, d_in):
    # Linear -> ReLU -> [BatchNorm] -> [Dropout], logit output
    layers, d = [], d_in
    for h in c["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if c["batchnorm"]:
            layers.append(nn.BatchNorm1d(h))
        if c["dropout"] > 0:
            layers.append(nn.Dropout(c["dropout"]))
        d = h
    model = nn.Sequential(*layers, nn.Linear(d, 1))
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def fit_predict(c, Xtr, ytr, Xte, seed):
    torch.manual_seed(seed)
    Xtr, ytr, Xte = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32), torch.tensor(Xte)
    model = build_mlp(c, Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(c["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(perm), c["batch_size"]):
            idx = perm[i:i + c["batch_size"]]
            if len(idx) < 2:          # batchnorm needs > 1 sample
                continue
            opt.zero_grad()
            loss_fn(model(Xtr[idx]).squeeze(-1), ytr[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xte).squeeze(-1).numpy()


def mlp(X, y, tr, te, seed=SEED, config=MLP_CONFIG):
    # fixed MLP: train on training genes, return held-out logits
    return fit_predict(config, X[tr], y[tr], X[te], seed)
