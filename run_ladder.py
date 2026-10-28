"""
Full adaptation methods benchmark: adaptation ladder x backbones x K.

Methods computed per run (all evaluated on the identical split):
  - zeroshot   : CLIP text classifier (fixed descriptive prompts)
  - proto      : visual prototypes (normalized mean of clean support feats)
  - combproto  : combined text-visual prototypes, parameter-free:
                 normalize(visual_prototype + text_embedding)
  - tip        : training-free Tip-Adapter (augmented-view cache keys)
  - tip_f      : Tip-Adapter-F (fine-tuned cache keys, random augmented
                 view per support image per epoch)

Backbones: see BACKBONES. K values: see K_LIST. N_RUNS seeds per config.

Reproducibility guarantees:
  - Split depends ONLY on (seed, sorted file list): identical across
    backbones and methods for the same (K, run).
  - Supports are NESTED across K for the same seed (the K=4 support is a
    subset of the K=16 support), since the per-class shuffle is identical
    and supports are taken as prefixes.
  - Clean and augmented features are extracted once per backbone and
    cached to disk with a verified manifest.
  - alpha/beta fixed to Tip-Adapter paper defaults (no dataset tuning).
  - Full config dumped per output directory.

Output layout:
  results/<pretrained>/features.pt, features_aug.pt (+ manifests)
  results/<pretrained>/k<K>/config.json
  results/<pretrained>/k<K>/runs/run_XX.npz
      keys: clip_logits, proto_logits, combproto_logits,
            cache_logits_tf, tip_logits, cache_logits_f, tip_f_logits,
            test_labels, test_idx, train_idx, seed

analyze_selective.py can be pointed at any results/<pretrained>/k<K> dir.
"""

import os
import gc
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import open_clip

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_ROOT = "crops"
CLASSES = [
    "Corrosion",
    "Detachment",
    "Cracks",
    "Stains",
    "Spalling",
]

BACKBONES = [
    ("ViT-H-14", "metaclip_fullcc"),
    ("ViT-H-14", "laion2b_s32b_b79k"),
]

K_LIST = [1, 2, 4, 8, 16]
N_RUNS = 30
SEED_BASE = 0

ALPHA = 1.0   # Tip-Adapter paper defaults (ImageNet 16-shot); not tuned here
BETA = 5.5

TRAIN_EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-2

N_AUG_VIEWS = 10          # augmented cache
AUG_SCALE = (0.5, 1.0)

RESULTS_ROOT = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_PROMPT_NAMES = {
    "Corrosion": "metal corrosion: material degradation, often associated with rust",
    "Detachment": "surface detachment: peeling or loss of surface layers such as plaster or coating",
    "Cracks": "a crack: thin linear fracture in walls or structural elements",
    "Stains": "damp stains: discoloration caused by moisture or water infiltration",
    "Spalling": "spalling: breaking or flaking of concrete or masonry surfaces",
}

TEMPLATES = [
    "a photo of {}.",
    "a photo of {} damage on a structure.",
    "a close-up photo of {}",
    "an image showing {}.",
]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageListDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img)


def list_dataset(data_root, classes, k_max):
    files = {}
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for c in classes:
        cdir = os.path.join(data_root, c)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"Class directory not found: {cdir}")
        paths = [os.path.join(cdir, f) for f in sorted(os.listdir(cdir))
                 if f.lower().endswith(valid_ext)]
        if len(paths) <= k_max:
            raise RuntimeError(f"Class '{c}' has {len(paths)} <= K_max={k_max}")
        files[c] = paths
    return files


# ----------------------------------------------------------------------
# Features (cached per backbone)
# ----------------------------------------------------------------------
@torch.no_grad()
def build_text_classifier(model, tokenizer):
    model.eval()
    weights = []
    for cls in CLASSES:
        texts = [t.format(CLASS_PROMPT_NAMES[cls]) for t in TEMPLATES]
        tf = F.normalize(model.encode_text(tokenizer(texts).to(DEVICE)), dim=-1)
        weights.append(F.normalize(tf.mean(dim=0), dim=0))
    return torch.stack(weights, dim=0)  # [C, d]


@torch.no_grad()
def encode_paths(model, preprocess, all_paths):
    dl = DataLoader(ImageListDataset(all_paths, preprocess),
                    batch_size=64, shuffle=False, num_workers=4)
    feats = []
    for imgs in dl:
        feats.append(F.normalize(model.encode_image(imgs.to(DEVICE)), dim=-1).cpu())
    return torch.cat(feats, dim=0)


def build_aug_preprocess(preprocess):
    from torchvision import transforms as T
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    crop = next(t for t in preprocess.transforms if isinstance(t, T.CenterCrop))
    size = crop.size[0] if isinstance(crop.size, (tuple, list)) else crop.size
    return T.Compose([
        T.RandomResizedCrop(size, scale=AUG_SCALE,
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        norm,
    ])


def cached(path_pt, path_json, manifest, build_fn):
    """Generic disk cache with manifest verification."""
    if os.path.exists(path_pt) and os.path.exists(path_json):
        with open(path_json) as f:
            if json.load(f) == manifest:
                print(f"  reusing cache {path_pt}")
                return torch.load(path_pt)
        print(f"  cache mismatch at {path_pt} -> rebuilding")
    out = build_fn()
    torch.save(out, path_pt)
    with open(path_json, "w") as f:
        json.dump(manifest, f)
    return out


def get_backbone_features(model, preprocess, all_paths, bdir, pretrained):
    man = {"paths": all_paths, "pretrained": pretrained}
    feats = cached(os.path.join(bdir, "features.pt"),
                   os.path.join(bdir, "features_manifest.json"), man,
                   lambda: (print("  extracting clean features ..."),
                            encode_paths(model, preprocess, all_paths))[1])
    aug = None
    if N_AUG_VIEWS > 0:
        man_aug = {**man, "n_views": N_AUG_VIEWS, "scale": list(AUG_SCALE)}

        def build_aug():
            aug_pre = build_aug_preprocess(preprocess)
            views = []
            for v in range(N_AUG_VIEWS):
                set_all_seeds(SEED_BASE * 1000 + v)
                views.append(encode_paths(model, aug_pre, all_paths))
                print(f"  augmented view {v + 1}/{N_AUG_VIEWS}")
            return torch.stack(views, dim=1)  # [N, V, d]

        aug = cached(os.path.join(bdir, "features_aug.pt"),
                     os.path.join(bdir, "features_aug_manifest.json"),
                     man_aug, build_aug)
    return feats, aug


# ----------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------
def cache_branch_logits(affinity, cache_values, beta):
    return ((-1) * (beta - beta * affinity)).exp() @ cache_values


class TipAdapterF(nn.Module):
    def __init__(self, cache_keys):
        super().__init__()
        self.adapter = nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False)
        self.adapter.weight.data = cache_keys.clone()

    def forward(self, x):
        return self.adapter(x)


def train_tip_adapter_f(train_view_feats, train_labels, clip_weights,
                        cache_keys, cache_values):
    adapter = TipAdapterF(cache_keys).to(DEVICE)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, eps=1e-4,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
    train_view_feats = train_view_feats.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    n, n_views, _ = train_view_feats.shape
    for _ in range(TRAIN_EPOCHS):
        adapter.train()
        vi = torch.randint(0, n_views, (n,), device=DEVICE)
        feats_epoch = train_view_feats[torch.arange(n, device=DEVICE), vi]
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            fb, lb = feats_epoch[idx], train_labels[idx]
            cl = cache_branch_logits(adapter(fb), cache_values, BETA)
            loss = F.cross_entropy(100.0 * fb @ clip_weights.T + ALPHA * cl, lb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    adapter.eval()
    return adapter


def run_all_methods(train_feats, train_views, train_labels,
                    test_feats, clip_weights, n_classes):
    """Computes logits for all 5 methods on one split. All tensors on DEVICE
    except train_feats/train_views (moved as needed). Returns dict of cpu
    numpy arrays."""
    out = {}
    test_feats = test_feats.to(DEVICE)

    with torch.no_grad():
        # zero-shot
        clip_logits = 100.0 * test_feats @ clip_weights.T
        out["clip_logits"] = clip_logits

        # visual prototypes (clean features)
        protos = []
        for c in range(n_classes):
            protos.append(F.normalize(
                train_feats[train_labels == c].mean(dim=0), dim=0))
        protos = torch.stack(protos, dim=0).to(DEVICE)          # [C, d]
        out["proto_logits"] = 100.0 * test_feats @ protos.T

        # combined text-visual prototypes (parameter-free)
        comb = F.normalize(protos + clip_weights, dim=-1)
        out["combproto_logits"] = 100.0 * test_feats @ comb.T

        # Tip-Adapter cache (augmented-mean keys if available)
        if train_views.shape[1] > 1:
            cache_keys = F.normalize(train_views.mean(dim=1), dim=-1)
        else:
            cache_keys = train_feats.clone()
        cache_values = F.one_hot(train_labels, num_classes=n_classes).float().to(DEVICE)

        affinity_tf = test_feats @ cache_keys.T.to(DEVICE)
        out["cache_logits_tf"] = cache_branch_logits(affinity_tf, cache_values, BETA)
        out["tip_logits"] = clip_logits + ALPHA * out["cache_logits_tf"]

    # Tip-Adapter-F (gradient part outside no_grad)
    adapter = train_tip_adapter_f(train_views, train_labels, clip_weights,
                                  cache_keys, cache_values)
    with torch.no_grad():
        out["cache_logits_f"] = cache_branch_logits(adapter(test_feats),
                                                    cache_values, BETA)
        out["tip_f_logits"] = clip_logits + ALPHA * out["cache_logits_f"]

    return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}


METHOD_LOGITS = {"zeroshot": "clip_logits", "proto": "proto_logits",
                 "combproto": "combproto_logits", "tip": "tip_logits",
                 "tip_f": "tip_f_logits"}


def metrics(logits, labels):
    from sklearn.metrics import f1_score, accuracy_score
    preds = logits.argmax(axis=1)
    return (accuracy_score(labels, preds) * 100,
            f1_score(labels, preds, average="macro") * 100)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    n_classes = len(CLASSES)
    files_by_class = list_dataset(DATA_ROOT, CLASSES, max(K_LIST))
    all_paths, all_labels = [], []
    idx_by_class = {}
    off = 0
    for ci, cls in enumerate(CLASSES):
        ps = files_by_class[cls]
        all_paths += ps
        all_labels += [ci] * len(ps)
        idx_by_class[cls] = list(range(off, off + len(ps)))
        off += len(ps)
        print(f"  {cls}: {len(ps)} images")
    all_labels_t = torch.tensor(all_labels)

    summary = {}  # (pretrained, K, method) -> (accs, f1s)

    for model_name, pretrained in BACKBONES:
        bdir = os.path.join(RESULTS_ROOT, pretrained)
        os.makedirs(bdir, exist_ok=True)
        print(f"\n########## {model_name} / {pretrained} ##########")
        set_all_seeds(SEED_BASE)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(DEVICE).eval()

        clip_weights = build_text_classifier(model, tokenizer).to(DEVICE)
        all_feats, aug_feats = get_backbone_features(
            model, preprocess, all_paths, bdir, pretrained)

        # image encoder no longer needed for this backbone
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for K in K_LIST:
            kdir = os.path.join(bdir, f"k{K}")
            os.makedirs(os.path.join(kdir, "runs"), exist_ok=True)
            config = {
                "model": model_name, "pretrained": pretrained,
                "k_shot": K, "n_runs": N_RUNS, "seed_base": SEED_BASE,
                "alpha": ALPHA, "beta": BETA, "train_epochs": TRAIN_EPOCHS,
                "lr": LR, "batch_size": BATCH_SIZE,
                "weight_decay": WEIGHT_DECAY,
                "n_aug_views": N_AUG_VIEWS, "aug_scale": list(AUG_SCALE),
                "classes": CLASSES, "class_prompts": CLASS_PROMPT_NAMES,
                "templates": TEMPLATES,
                "n_images_per_class": {c: len(files_by_class[c]) for c in CLASSES},
                "methods": list(METHOD_LOGITS),
                "note": "Splits depend only on (seed, sorted file list); "
                        "supports nested across K for equal seeds.",
            }
            with open(os.path.join(kdir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            accs = {m: [] for m in METHOD_LOGITS}
            f1s = {m: [] for m in METHOD_LOGITS}

            for run in range(N_RUNS):
                seed = SEED_BASE + run
                set_all_seeds(seed)
                rng = random.Random(seed)

                train_idx, test_idx = [], []
                for cls in CLASSES:
                    idxs = idx_by_class[cls][:]
                    rng.shuffle(idxs)
                    train_idx += idxs[:K]        # prefix -> nested across K
                    test_idx += idxs[K:]
                train_idx_t = torch.tensor(train_idx)
                test_idx_t = torch.tensor(test_idx)

                train_feats = all_feats[train_idx_t]
                train_labels = all_labels_t[train_idx_t]
                train_views = (aug_feats[train_idx_t] if aug_feats is not None
                               else train_feats.unsqueeze(1))
                test_labels = all_labels_t[test_idx_t]

                logits = run_all_methods(train_feats, train_views, train_labels,
                                         all_feats[test_idx_t], clip_weights,
                                         n_classes)

                np.savez_compressed(
                    os.path.join(kdir, "runs", f"run_{run:02d}.npz"),
                    **logits,
                    test_labels=test_labels.numpy().astype(np.int64),
                    test_idx=test_idx_t.numpy().astype(np.int64),
                    train_idx=train_idx_t.numpy().astype(np.int64),
                    seed=np.int64(seed))

                labels_np = test_labels.numpy()
                for m, key in METHOD_LOGITS.items():
                    a, ff = metrics(logits[key], labels_np)
                    accs[m].append(a)
                    f1s[m].append(ff)
                if (run + 1) % 10 == 0 or run == N_RUNS - 1:
                    print(f"  K={K:2d} run {run + 1}/{N_RUNS}  "
                          f"tip_f acc so far: {np.mean(accs['tip_f']):.2f}")

            for m in METHOD_LOGITS:
                summary[(pretrained, K, m)] = (np.array(accs[m]), np.array(f1s[m]))

        del all_feats, aug_feats, clip_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Final compact report
    # ------------------------------------------------------------------
    kmax = max(K_LIST)
    print(f"\n================ MAIN TABLE (K={kmax}, mean +- std over "
          f"{N_RUNS} runs) ================")
    print(f"{'method':12s}" + "".join(f"  {p[:18]:>25s}" for _, p in BACKBONES))
    for m in METHOD_LOGITS:
        row = f"{m:12s}"
        for _, p in BACKBONES:
            a, ff = summary[(p, kmax, m)]
            row += f"  {a.mean():5.2f}+-{a.std():4.2f}/{ff.mean():5.2f}+-{ff.std():4.2f}"
        print(row + "   (acc/mF1)")

    print("\n================ K SWEEP, Tip-Adapter-F (acc / macro-F1) ================")
    header = f"{'K':>4s}" + "".join(f"  {p[:18]:>25s}" for _, p in BACKBONES)
    print(header)
    for K in K_LIST:
        row = f"{K:4d}"
        for _, p in BACKBONES:
            a, ff = summary[(p, K, "tip_f")]
            row += f"  {a.mean():5.2f}+-{a.std():4.2f}/{ff.mean():5.2f}+-{ff.std():4.2f}"
        print(row)

    # dump summary json
    out = {f"{p}|k{K}|{m}": {"acc_mean": float(a.mean()), "acc_std": float(a.std()),
                             "f1_mean": float(ff.mean()), "f1_std": float(ff.std())}
           for (p, K, m), (a, ff) in summary.items()}
    with open(os.path.join(RESULTS_ROOT, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary saved to {RESULTS_ROOT}/summary.json")
    print("Per-run logits in results/<pretrained>/k<K>/runs/ "
          "(point analyze_selective.py at any of those directories).")


if __name__ == "__main__":
    main()