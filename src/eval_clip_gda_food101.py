import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import clip


def load_optional_group_info(class_names):
    group_map = None
    group_alphas = None

    try:
        import confusion_groups
        if hasattr(confusion_groups, "difficulty_groups"):
            difficulty_groups = confusion_groups.difficulty_groups
            tmp = {}

            if isinstance(difficulty_groups, dict):
                for gname, vals in difficulty_groups.items():
                    idxs = []
                    for v in vals:
                        if isinstance(v, int):
                            if 0 <= v < len(class_names):
                                idxs.append(v)
                        elif isinstance(v, str):
                            if v in class_names:
                                idxs.append(class_names.index(v))
                    tmp[gname] = idxs

            group_map = tmp
            print("Loaded difficulty_groups from confusion_groups.py")
    except Exception as e:
        print(f"[WARN] Could not load confusion_groups.py groups: {e}")

    try:
        import group_weights
        if hasattr(group_weights, "group_weights"):
            gw = group_weights.group_weights
            if isinstance(gw, dict):
                group_alphas = {}
                for k, v in gw.items():
                    try:
                        group_alphas[k] = float(v)
                    except Exception:
                        pass
                print("Loaded group_weights from group_weights.py")
    except Exception as e:
        print(f"[WARN] Could not load group_weights.py weights: {e}")

    return group_map, group_alphas


def build_fallback_groups_from_confusion_csv(class_names):
    other = list(range(len(class_names)))
    return {
        "A": [],
        "B": [],
        "C": [],
        "Other": other,
    }


def normalize_group_map(group_map, class_names):
    if group_map is None:
        return build_fallback_groups_from_confusion_csv(class_names)

    out = {}
    used = set()

    for key in ["A", "B", "C"]:
        idxs = group_map.get(key, [])
        idxs = [i for i in idxs if 0 <= i < len(class_names)]
        out[key] = idxs
        used.update(idxs)

    out["Other"] = [i for i in range(len(class_names)) if i not in used]
    return out


def build_group_alpha_vector(class_names, group_map, loaded_group_alphas=None):
    group_map = normalize_group_map(group_map, class_names)

    default_group_alphas = {
        "A": 0.45,
        "B": 0.50,
        "C": 0.55,
        "Other": 0.50,
    }

    if loaded_group_alphas is not None:
        for k in default_group_alphas:
            if k in loaded_group_alphas:
                default_group_alphas[k] = loaded_group_alphas[k]

    alpha_vec = np.zeros(len(class_names), dtype=np.float32)

    for gname, idxs in group_map.items():
        alpha = float(default_group_alphas.get(gname, 0.50))
        for i in idxs:
            alpha_vec[i] = alpha

    return alpha_vec, group_map, default_group_alphas


def sample_fewshot_indices_by_class(dataset, shot_per_class=5, seed=42):
    """Sample shot_per_class indices for each class from the dataset."""
    rng = np.random.default_rng(seed)

    # Build class -> indices mapping
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset._labels):
        class_to_indices[label].append(idx)

    # Sample shot_per_class indices from each class
    support_indices = []
    for c in sorted(class_to_indices.keys()):
        indices = class_to_indices[c]
        if len(indices) >= shot_per_class:
            chosen = rng.choice(indices, size=shot_per_class, replace=False)
        else:
            chosen = indices
        support_indices.extend(chosen)

    return support_indices


def compute_class_means(model, support_loader, num_classes, device):
    feat_dim = None
    class_sums = None
    class_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for images, labels in support_loader:
            images = images.to(device)
            labels = labels.to(device)

            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            if feat_dim is None:
                feat_dim = feats.shape[1]
                class_sums = torch.zeros(num_classes, feat_dim, dtype=feats.dtype, device=device)

            for i in range(labels.size(0)):
                y = labels[i].item()
                class_sums[y] += feats[i]
                class_counts[y] += 1

    class_means = torch.zeros_like(class_sums)
    for c in range(num_classes):
        if class_counts[c] > 0:
            class_means[c] = class_sums[c] / class_counts[c]
            class_means[c] = class_means[c] / class_means[c].norm()

    return class_means


def build_text_features(model, class_names, device):
    prompts = [f"a photo of {name.replace('_', ' ')}" for name in class_names]
    tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def compute_per_class_accuracy(y_true, y_pred, num_classes):
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1

    acc = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        acc[c] = (correct[c] / total[c]) if total[c] > 0 else np.nan
    return acc, correct, total


def evaluate_global_alpha(model, text_features, class_means, test_loader, alpha, device):
    all_true = []
    all_pred = []

    total = 0
    correct_top1 = 0
    correct_top5 = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            image_logits = image_features @ class_means.T
            text_logits = image_features @ text_features.T
            logits = (1.0 - alpha) * image_logits + alpha * text_logits

            _, top5 = logits.topk(5, dim=1)
            pred1 = top5[:, 0]

            correct_top1 += (pred1 == labels).sum().item()
            correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

            all_true.append(labels.cpu().numpy())
            all_pred.append(pred1.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    per_class_acc, class_correct, class_total = compute_per_class_accuracy(
        y_true, y_pred, text_features.shape[0]
    )

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total
    return top1, top5, per_class_acc, class_correct, class_total


def evaluate_group_alpha(model, text_features, class_means, test_loader, alpha_vec, group_map, device):
    """
    Evaluate using group-specific alphas with PROPER handling:
    For each sample, use the alpha corresponding to its TRUE class.
    This tests: "If we knew which group a sample belonged to, would group-specific alpha help?"
    """
    all_true = []
    all_pred = []

    total = 0
    correct_top1 = 0
    correct_top5 = 0

    # Build label -> alpha mapping
    alpha_vec_t = torch.tensor(alpha_vec, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            image_logits = image_features @ class_means.T  # [B, C]
            text_logits = image_features @ text_features.T  # [B, C]

            # For each sample, use alpha based on its TRUE label
            # This gives us the "oracle" upper bound for group-specific alpha
            batch_alphas = alpha_vec_t[labels]  # [B]
            batch_alphas = batch_alphas.unsqueeze(1)  # [B, 1]

            # Apply same alpha across all classes for each sample
            logits = (1.0 - batch_alphas) * image_logits + batch_alphas * text_logits

            _, top5 = logits.topk(5, dim=1)
            pred1 = top5[:, 0]

            correct_top1 += (pred1 == labels).sum().item()
            correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

            all_true.append(labels.cpu().numpy())
            all_pred.append(pred1.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    per_class_acc, class_correct, class_total = compute_per_class_accuracy(
        y_true, y_pred, text_features.shape[0]
    )

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total
    return top1, top5, per_class_acc, class_correct, class_total


def mean_group_acc(per_class_acc, idxs):
    vals = [per_class_acc[i] for i in idxs if not np.isnan(per_class_acc[i])]
    if len(vals) == 0:
        return float("nan")
    return 100.0 * float(np.mean(vals))


def save_per_class_csv(out_dir, class_names, global_acc, group_acc, global_correct, global_total, group_correct, group_total):
    out_path = out_dir / "clip_gda_per_class_accuracy.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("class,global_correct,global_total,global_acc,group_correct,group_total,group_acc\n")
        for i, name in enumerate(class_names):
            f.write(
                f"{name},{int(global_correct[i])},{int(global_total[i])},{float(global_acc[i]):.6f},"
                f"{int(group_correct[i])},{int(group_total[i])},{float(group_acc[i]):.6f}\n"
            )
    print(f"Saved per-class GDA results to {out_path}")


def main():
    data_root = Path("../data/food-101")
    output_dir = Path("../results/clip_gda_food101_vit_b_32_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    shot_per_class = 5
    batch_size = 64
    num_workers = 4
    global_alpha = 0.50
    model_name = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP model {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    print("Loading Food-101 train split for support set...")
    train_dataset = datasets.Food101(
        root=str(data_root),
        split="train",
        transform=preprocess,
        download=True,
    )

    print("Loading Food-101 test split...")
    test_dataset = datasets.Food101(
        root=str(data_root),
        split="test",
        transform=preprocess,
        download=True,
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    assert num_classes == 101, f"Expected 101 classes, got {num_classes}"

    print("Building 5-shot support indices per class...")
    support_indices = sample_fewshot_indices_by_class(train_dataset, shot_per_class=shot_per_class)
    support_subset = Subset(train_dataset, support_indices)

    support_loader = DataLoader(
        support_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Encoding support images and computing class mean features...")
    class_means = compute_class_means(model, support_loader, num_classes, device)
    print(f"class_means shape: {tuple(class_means.shape)}")

    print("Building CLIP text classifier for Food-101 classes...")
    text_features = build_text_features(model, class_names, device)
    print(f"text_features shape: {tuple(text_features.shape)}")

    group_map_loaded, loaded_group_alphas = load_optional_group_info(class_names)
    alpha_vec, group_map, alpha_used = build_group_alpha_vector(
        class_names, group_map_loaded, loaded_group_alphas
    )

    print(f"Using group alphas: {alpha_used}")
    print(f"Group sizes: A={len(group_map['A'])}, B={len(group_map['B'])}, C={len(group_map['C'])}, Other={len(group_map['Other'])}")

    print(f"\nEvaluating CLIP+GDA with global alpha={global_alpha:.3f}...")
    start = time.time()
    global_top1, global_top5, global_per_class, global_correct, global_total = evaluate_global_alpha(
        model, text_features, class_means, test_loader, global_alpha, device
    )
    print(f"Global-alpha CLIP+GDA Top-1 accuracy: {global_top1:.2f}%")
    print(f"Global-alpha CLIP+GDA Top-5 accuracy: {global_top5:.2f}%")
    print(f"Global-alpha eval time: {(time.time() - start)/60:.2f} min")

    print("\nEvaluating CLIP+GDA with group-specific alphas (oracle)...")
    print("(Each sample uses alpha based on its TRUE class group)")
    start = time.time()
    group_top1, group_top5, group_per_class, group_correct, group_total = evaluate_group_alpha(
        model, text_features, class_means, test_loader, alpha_vec, group_map, device
    )
    print(f"Group-alpha CLIP+GDA Top-1 accuracy: {group_top1:.2f}%")
    print(f"Group-alpha CLIP+GDA Top-5 accuracy: {group_top5:.2f}%")
    print(f"Group-alpha eval time: {(time.time() - start)/60:.2f} min")

    print("\n" + "="*60)
    print("Per-group Top-1 accuracy (Global alpha vs Group alpha):")
    print("="*60)
    for g in ["A", "B", "C", "Other"]:
        idxs = group_map.get(g, [])
        g_global = mean_group_acc(global_per_class, idxs)
        g_group = mean_group_acc(group_per_class, idxs)
        alpha_for_group = alpha_used.get(g, 0.50)
        delta = g_group - g_global
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"Group {g} (α={alpha_for_group:.2f}): Global={g_global:.2f}%  Group={g_group:.2f}%  [{arrow} {abs(delta):.2f}%]")

    print("="*60)
    overall_delta = group_top1 - global_top1
    arrow = "↑" if overall_delta > 0 else "↓" if overall_delta < 0 else "→"
    print(f"OVERALL: Global={global_top1:.2f}%  Group={group_top1:.2f}%  [{arrow} {abs(overall_delta):.2f}%]")
    print("="*60)

    save_per_class_csv(
        output_dir,
        class_names,
        global_per_class,
        group_per_class,
        global_correct,
        global_total,
        group_correct,
        group_total,
    )

    print("\nDone: global vs group-alpha CLIP+GDA comparison complete.")


if __name__ == "__main__":
    main()