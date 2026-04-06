import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import clip


def eval_clip_food101(
    data_root: str = "../data/food-101",
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 4,
    model_name: str = "ViT-B/32",
    device: str | None = None,
    output_dir: str = "../results",
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP model and preprocess
    print(f"Loading CLIP model {model_name} on {device}...")
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    # Food-101 dataset (torchvision).[web:224][web:226]
    print(f"Loading Food-101 {split} split from {data_root}...")
    dataset = datasets.Food101(
        root=str(data_root),
        split=split,
        transform=preprocess,
        download=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = dataset.classes  # 101 class names.[web:226]
    num_classes = len(class_names)
    assert num_classes == 101, f"Expected 101 classes, got {num_classes}"

    # Zero-shot text prompts
    print("Preparing text prompts...")
    with torch.no_grad():
        text_tokens = torch.cat(
            [
                clip.tokenize(f"a photo of {name.replace('_', ' ')}")
                for name in class_names
            ]
        ).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Stats
    correct_top1 = 0
    total = 0

    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    # Confusion matrix: rows=true, cols=pred
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    print("Evaluating CLIP on Food-101...")
    start = time.time()

    with torch.no_grad():
        for images_pre, labels in loader:
            images_pre = images_pre.to(device)
            labels = labels.to(device)

            # Image features
            image_features = model.encode_image(images_pre)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            # Similarity and predictions
            logits = 100.0 * image_features @ text_features.T
            preds = logits.argmax(dim=-1)

            correct = (preds == labels)
            correct_top1 += correct.sum().item()
            total += labels.size(0)

            # Per-class accuracy counts
            for i in range(labels.size(0)):
                cls = labels[i].item()
                class_total[cls] += 1
                if correct[i]:
                    class_correct[cls] += 1

            # Confusion matrix counts
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    elapsed = time.time() - start
    overall_acc = correct_top1 / total if total > 0 else 0.0

    print(f"Done in {elapsed/60:.2f} minutes.")
    print(f"Top-1 accuracy: {overall_acc * 100:.2f}%")

    # Output paths
    out_path = output_dir / f"clip_food101_{model_name.replace('/', '_')}_{split}"
    out_path.mkdir(parents=True, exist_ok=True)

    # Save overall stats
    stats_path = out_path / "stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"split: {split}\n")
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"total_images: {total}\n")
        f.write(f"top1_accuracy: {overall_acc:.6f}\n")
        f.write(f"elapsed_seconds: {elapsed:.2f}\n")

    print(f"Saved overall stats to {stats_path}")

    # Save per-class accuracy
    csv_path = out_path / "per_class_accuracy.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("class,correct,total,accuracy\n")
        for idx, name in enumerate(class_names):
            c = class_correct[idx].item()
            t = class_total[idx].item()
            a = c / t if t > 0 else 0.0
            f.write(f"{name},{c},{t},{a:.6f}\n")

    print(f"Saved per-class accuracy to {csv_path}")

    # Save confusion matrix as CSV (row = true, col = predicted)
    cm_path = out_path / "confusion_matrix.csv"
    with cm_path.open("w", encoding="utf-8") as f:
        # header row: empty cell + class names
        f.write("," + ",".join(class_names) + "\n")
        for i, row_name in enumerate(class_names):
            row = ",".join(str(x) for x in confusion[i].tolist())
            f.write(f"{row_name},{row}\n")

    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    eval_clip_food101()
