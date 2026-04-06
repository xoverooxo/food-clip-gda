"""
Baseline 1: Zero-shot CLIP on Food-101
Pure text-based classification - no few-shot support, no GDA
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import clip


def build_text_features(model, class_names, device):
    """Build text classifier from class names using prompt template."""
    prompts = [f"a photo of {name.replace('_', ' ')}" for name in class_names]
    tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def compute_per_class_accuracy(y_true, y_pred, num_classes):
    """Compute per-class accuracy statistics."""
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


def evaluate_zeroshot(model, text_features, test_loader, device):
    """Evaluate zero-shot CLIP (text-only classification)."""
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

            # Encode images
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Zero-shot: just compare to text features
            logits = image_features @ text_features.T

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
    return top1, top5, per_class_acc, class_correct, class_total, y_true, y_pred


def load_group_info(class_names):
    """Load confusion groups if available."""
    group_map = {"A": [], "B": [], "C": [], "Other": list(range(len(class_names)))}
    
    try:
        import confusion_groups
        if hasattr(confusion_groups, "difficulty_groups"):
            dg = confusion_groups.difficulty_groups
            used = set()
            for key in ["A", "B", "C"]:
                idxs = dg.get(key, [])
                valid = []
                for v in idxs:
                    if isinstance(v, int) and 0 <= v < len(class_names):
                        valid.append(v)
                        used.add(v)
                group_map[key] = valid
            group_map["Other"] = [i for i in range(len(class_names)) if i not in used]
            print("Loaded difficulty groups from confusion_groups.py")
    except Exception as e:
        print(f"[WARN] Could not load confusion_groups: {e}")
    
    return group_map


def mean_group_acc(per_class_acc, idxs):
    """Compute mean accuracy for a group of class indices."""
    vals = [per_class_acc[i] for i in idxs if not np.isnan(per_class_acc[i])]
    if len(vals) == 0:
        return float("nan")
    return 100.0 * float(np.mean(vals))


def save_results(output_dir, class_names, per_class_acc, class_correct, class_total):
    """Save per-class results to CSV."""
    out_path = output_dir / "clip_zeroshot_per_class_accuracy.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("class,correct,total,accuracy\n")
        for i, name in enumerate(class_names):
            f.write(f"{name},{int(class_correct[i])},{int(class_total[i])},{float(per_class_acc[i]):.6f}\n")
    print(f"Saved per-class results to {out_path}")


def save_confusion_matrix(output_dir, y_true, y_pred, class_names):
    """Save confusion matrix to CSV."""
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    out_path = output_dir / "clip_zeroshot_confusion_matrix.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, name in enumerate(class_names):
            row = ",".join(str(cm[i, j]) for j in range(num_classes))
            f.write(f"{name},{row}\n")
    print(f"Saved confusion matrix to {out_path}")


def main():
    data_root = Path("../data/food-101")
    output_dir = Path("../results/clip_zeroshot_food101")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    num_workers = 4
    model_name = "ViT-B/32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP model {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    print("Loading Food-101 test split...")
    test_dataset = datasets.Food101(
        root=str(data_root),
        split="test",
        transform=preprocess,
        download=True,
    )

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Building CLIP text classifier...")
    text_features = build_text_features(model, class_names, device)
    print(f"text_features shape: {tuple(text_features.shape)}")

    # Load group info for per-group analysis
    group_map = load_group_info(class_names)

    print("\nEvaluating Zero-shot CLIP on Food-101...")
    start = time.time()
    top1, top5, per_class_acc, class_correct, class_total, y_true, y_pred = evaluate_zeroshot(
        model, text_features, test_loader, device
    )
    eval_time = time.time() - start

    print("\n" + "="*60)
    print("ZERO-SHOT CLIP RESULTS")
    print("="*60)
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
    print(f"Evaluation time: {eval_time/60:.2f} min")
    print("="*60)

    print("\nPer-group accuracy:")
    print("-"*40)
    for g in ["A", "B", "C", "Other"]:
        idxs = group_map.get(g, [])
        g_acc = mean_group_acc(per_class_acc, idxs)
        print(f"Group {g} ({len(idxs):2d} classes): {g_acc:.2f}%")
    print("-"*40)

    # Find worst performing classes
    print("\nBottom 10 classes (lowest accuracy):")
    print("-"*40)
    sorted_idx = np.argsort(per_class_acc)
    for i in sorted_idx[:10]:
        print(f"  {class_names[i]}: {100*per_class_acc[i]:.1f}% ({class_correct[i]}/{class_total[i]})")

    # Find best performing classes
    print("\nTop 10 classes (highest accuracy):")
    print("-"*40)
    for i in sorted_idx[-10:][::-1]:
        print(f"  {class_names[i]}: {100*per_class_acc[i]:.1f}% ({class_correct[i]}/{class_total[i]})")

    # Save results
    save_results(output_dir, class_names, per_class_acc, class_correct, class_total)
    save_confusion_matrix(output_dir, y_true, y_pred, class_names)

    print("\nDone: Zero-shot CLIP evaluation complete.")


if __name__ == "__main__":
    main()