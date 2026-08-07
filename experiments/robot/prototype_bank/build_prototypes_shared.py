
import argparse
import time

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from experiments.robot.prototype_bank.config import (
    EMBED_DIM,
    K_PROTOTYPES,
    KMEANS_BATCH,
    KMEANS_ITER,
    KMEANS_N_INIT,
    KMEANS_SEED,
    L2_NORMALIZE_BEFORE_KMEANS,
    PROTOTYPE_BANK_DIR,
    SUITES,
    SUITE_TO_IDX,
    TARGET_LAYERS,
    TOKENS_DIR,
    TOWER,
)


def _l2_normalize(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def load_shared_layer_tokens(layer):

    key = f"arr_L{layer}"
    chunks, ids, per_suite_n = [], [], {}
    for suite in SUITES:
        path = TOKENS_DIR / f"{suite}_dinov2.npz"
        if not path.exists():
            print(f"  [skip] {path} missing")
            continue
        with np.load(path) as f:
            if key not in f.files:
                print(f"  [skip] {path}: key {key} missing")
                continue
            layer_tokens = f[key]
            if layer_tokens.shape[0] == 0:
                continue
            flattened_tokens = layer_tokens.reshape(-1, EMBED_DIM).astype(np.float32)
            chunks.append(flattened_tokens)
            ids.append(np.full(
                flattened_tokens.shape[0], SUITE_TO_IDX[suite], dtype=np.int64
            ))
            per_suite_n[suite] = (
                int(layer_tokens.shape[0]), int(flattened_tokens.shape[0])
            )
            print(
                f"  {suite:14s}  frames={layer_tokens.shape[0]:5d}  "
                f"tokens={flattened_tokens.shape[0]:8d}"
            )
    if not chunks:
        raise RuntimeError(f"layer L{layer}: no tokens found across suites.")
    return np.concatenate(chunks, axis=0), np.concatenate(ids, axis=0), per_suite_n


def build_shared_bank_for_layer(layer, k=K_PROTOTYPES):
    print(f"\n=== building SHARED {TOWER} L{layer} bank ===")
    tokens, suite_id, per_suite_n = load_shared_layer_tokens(layer)
    num_tokens, embedding_dim = tokens.shape
    print(f"  total tokens: {num_tokens:,}  dim={embedding_dim}")

    if L2_NORMALIZE_BEFORE_KMEANS:
        tokens_for_kmeans = _l2_normalize(tokens)
    else:
        tokens_for_kmeans = tokens

    print(f"  fitting MiniBatchKMeans(K={k}, batch={KMEANS_BATCH}, max_iter={KMEANS_ITER}) ...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=KMEANS_BATCH,
        max_iter=KMEANS_ITER,
        random_state=KMEANS_SEED,
        n_init=KMEANS_N_INIT,
        verbose=0,
    )
    assignments = kmeans.fit_predict(tokens_for_kmeans)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    print(f"  KMeans done in {time.time()-t0:.1f}s")

    if L2_NORMALIZE_BEFORE_KMEANS:
        centroids = _l2_normalize(centroids)


    suite_dist = np.zeros((k, len(SUITES)), dtype=np.float32)
    occupancy  = np.zeros(k, dtype=np.int64)
    for c in range(k):
        mask = (assignments == c)
        n_c  = int(mask.sum())
        occupancy[c] = n_c
        if n_c == 0:
            continue
        for s_idx in range(len(SUITES)):
            suite_dist[c, s_idx] = float((suite_id[mask] == s_idx).sum()) / n_c
    suite_labels = suite_dist.argmax(axis=1).astype(np.int64)

    empty = int((occupancy == 0).sum())
    print(f"  cluster occupancy:  min={occupancy.min()}  median={int(np.median(occupancy))}  "
          f"max={occupancy.max()}  empty={empty}")


    suite_label_counts = {SUITES[i]: int((suite_labels == i).sum()) for i in range(len(SUITES))}
    print(f"  dominant-suite counts: {suite_label_counts}")

    bank = {
        "centroids":    torch.from_numpy(centroids),
        "suite_labels": torch.from_numpy(suite_labels),
        "suite_dist":   torch.from_numpy(suite_dist),
        "occupancy":    torch.from_numpy(occupancy),
        "meta": {
            "tower":         TOWER,
            "layer":         int(layer),
            "K":             int(k),
            "embed_dim":     int(embedding_dim),
            "suites":        list(SUITES),
            "l2_normalized": bool(L2_NORMALIZE_BEFORE_KMEANS),
            "n_tokens":      int(num_tokens),
            "per_suite":     {s: {"n_frames": nf, "n_tokens": nt}
                              for s, (nf, nt) in per_suite_n.items()},
            "kmeans": {
                "batch_size": KMEANS_BATCH,
                "max_iter":   KMEANS_ITER,
                "seed":       KMEANS_SEED,
            },
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    out_path = PROTOTYPE_BANK_DIR / f"dinov2_L{layer}.pt"
    torch.save(bank, out_path)
    print(f"  saved {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
    return bank


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a SHARED (fused-across-suites) DinoV2 prototype bank per layer.")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Subset of layers (default = final DRL layers L18-22).")
    p.add_argument("--k", type=int, default=K_PROTOTYPES,
                   help="K clusters per layer.")
    return p.parse_args()


def main():
    args   = parse_args()
    layers = args.layers if args.layers else TARGET_LAYERS

    for layer in layers:
        build_shared_bank_for_layer(layer, k=args.k)


if __name__ == "__main__":
    main()
