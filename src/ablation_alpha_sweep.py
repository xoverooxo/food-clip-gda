"""
Ablation Study: Alpha Sweep for CLIP+GDA on Food-101
"""

import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import clip


def sample_fewshot_indices_by_class(dataset, shot_per_class=5, seed=42):
    rng = np.random.default_rng(seed)
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset._labels):
        class_to_indices[label].append(idx)

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


def evaluate_alpha(model, text_features, class_means, test_loader, alpha, device):
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

            # alpha=0 -> pure GDA, alpha=1 -> pure CLIP text
            logits = (1.0 - alpha) * image_logits + alpha * text_logits

            _, top5 = logits.topk(5, dim=1)
            pred1 = top5[:, 0]

            correct_top1 += (pred1 == labels).sum().item()
            correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total
    return top1, top5


def main():
    data_root = Path("../data/food-101")
    output_dir = Path("../results/ablation_alpha_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    shot_per_class = 5
    batch_size = 64
    num_workers = 4
    model_name = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"Loading CLIP model {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    print("Loading Food-101 train split for support set...")
    train_dataset = datasets.Food101(root=str(data_root), split="train", transform=preprocess, download=True)

    print("Loading Food-101 test split...")
    test_dataset = datasets.Food101(root=str(data_root), split="test", transform=preprocess, download=True)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Building 5-shot support set...")
    support_indices = sample_fewshot_indices_by_class(train_dataset, shot_per_class=shot_per_class)
    support_subset = Subset(train_dataset, support_indices)

    support_loader = DataLoader(support_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print("Computing class means from support set...")
    class_means = compute_class_means(model, support_loader, num_classes, device)

    print("Building CLIP text classifier...")
    text_features = build_text_features(model, class_names, device)

    print("\n" + "="*70)
    print("ABLATION STUDY: Alpha Sweep for CLIP+GDA on Food-101")
    print("="*70)
    print("alpha=0.0 -> pure GDA (image-based few-shot)")
    print("alpha=1.0 -> pure CLIP text (zero-shot)")
    print("="*70)

    results = []
    for alpha in alphas:
        print(f"\nEvaluating alpha={alpha:.1f}...", end=" ", flush=True)
        start = time.time()
        top1, top5 = evaluate_alpha(model, text_features, class_means, test_loader, alpha, device)
        elapsed = time.time() - start
        print(f"Top-1: {top1:.2f}%, Top-5: {top5:.2f}% ({elapsed:.1f}s)")
        results.append((alpha, top1, top5))

    # Print summary
    print("\n" + "="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    print(f"{'Alpha':<10} {'Top-1':<12} {'Top-5':<12} {'Note'}")
    print("-"*70)

    best_top1 = max(r[1] for r in results)
    for alpha, top1, top5 in results:
        if alpha == 0.0:
            note = "Pure GDA"
        elif alpha == 1.0:
            note = "Pure CLIP text"
        elif alpha == 0.5:
            note = "Baseline"
        else:
            note = ""
        marker = " ** BEST" if top1 == best_top1 else ""
        print(f"{alpha:<10.1f} {top1:<12.2f} {top5:<12.2f} {note}{marker}")

    print("="*70)

    # Key findings
    pure_gda = results[0][1]
    pure_clip = results[-1][1]
    baseline = results[5][1]

    print("\nKEY FINDINGS:")
    print(f"  Pure GDA (a=0.0):   {pure_gda:.2f}%")
    print(f"  Baseline (a=0.5):   {baseline:.2f}%")
    print(f"  Pure CLIP (a=1.0):  {pure_clip:.2f}%")

    if pure_clip > baseline:
        print(f"\n  >> CLIP alone beats GDA+CLIP by {pure_clip - baseline:.2f}%")
        print("  >> GDA HURTS on fine-grained Food-101!")

    # Save CSV
    csv_path = output_dir / "alpha_sweep_results.csv"
    with csv_path.open("w") as f:
        f.write("alpha,top1,top5\n")
        for alpha, top1, top5 in results:
            f.write(f"{alpha:.1f},{top1:.4f},{top5:.4f}\n")
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()