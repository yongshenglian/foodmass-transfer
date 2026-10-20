"""USDA FNDDS lookup features for each image (DietAI24-style grounding, no labels used).

Query = the food names Qwen wrote in its blind answer; documents = FNDDS 2021-2023 main food descriptions.
Both embedded with bge-small; top-3 foods by cosine similarity, similarity-weighted. Features per image:
log grams per cup, log median standard-portion weight, log smallest/largest listed portion, top similarity.
Caveat: ASA24 labels come from FNDDS 2011-2012 portion weights, many unchanged in 2021-2023, so this feature is
partly circular on ASA24 and must be confirmed on weighed-food data. Run from code/.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

A = Path("../experiments/asa24_external")
ENC = Path.home() / "models/bge-small-en-v1.5"


def encode(texts, tok, m, prefix=""):
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), 256):
            x = tok([prefix + t for t in texts[s:s + 256]], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
            out.append(torch.nn.functional.normalize(m(**x).last_hidden_state[:, 0], dim=1).cpu().numpy())
    return np.concatenate(out)


def fndds_table():
    d = pd.read_excel(Path.home() / "data/fndds/pw_2021-2023.xlsx", header=1)
    d.columns = ["food", "desc", "wweia", "wweia_desc", "seq", "portion", "grams"]
    d = d[d.grams > 0]
    cup = d[d.portion.str.lower().str.match(r"^1 cup\b", na=False)].groupby("food").grams.median()
    g = d.groupby("food")
    t = pd.DataFrame({"desc": g.desc.first(), "median_portion": g.grams.median(), "min_portion": g.grams.min(),
                      "max_portion": g.grams.max()})
    t["gcup"] = cup
    return t


def main():
    t = fndds_table()
    emb = np.load(A / "food_name_embeddings.npz", allow_pickle=True)
    ids, text = emb["dish_ids"].astype(str), emb["text"].astype(str)
    tok = AutoTokenizer.from_pretrained(ENC); m = AutoModel.from_pretrained(ENC).cuda().eval()
    doc = encode(list(t.desc), tok, m)
    q = encode([x.split(",")[0] for x in text], tok, m, prefix="Represent this sentence for searching relevant passages: ")
    sim = q @ doc.T
    top = np.argsort(-sim, 1)[:, :3]; s = np.take_along_axis(sim, top, 1); w = np.exp((s - s[:, :1]) / 0.02); w /= w.sum(1, keepdims=True)
    feats = {}
    gcup_all = np.log(t.gcup.fillna(t.median_portion).to_numpy())
    for name, col in [("log_gcup", gcup_all), ("log_median_portion", np.log(t.median_portion.to_numpy())),
                      ("log_min_portion", np.log(t.min_portion.to_numpy())), ("log_max_portion", np.log(t.max_portion.to_numpy()))]:
        feats[name] = (col[top] * w).sum(1)
    out = pd.DataFrame({"dish_id": ids, "query": [x.split(",")[0] for x in text], "match": t.desc.to_numpy()[top[:, 0]],
                        "top_similarity": s[:, 0], **feats})
    out.to_csv(A / "fndds_lookup_features.csv", index=False)
    print(out.sample(8, random_state=1)[["query", "match", "top_similarity", "log_gcup"]].to_string())


if __name__ == "__main__":
    main()
