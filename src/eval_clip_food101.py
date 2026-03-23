import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101

from clip_food101_baseline import load_clip_model


@torch.no_grad()
def eval_clip_food101(batch_size: int = 64, root: str = "../data", out_dir: str = "../results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess, tokenizer = load_clip_model()
    model.to(device)
    model.eval()

    dataset = Food101(root=root, split="test")
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Loaded Food-101 test split with {len(dataset)} images and {num_classes} classes")

    def collate_fn(batch):
        images, labels = zip(*batch)
        images_pre = torch.stack([preprocess(img) for img in images])
        labels = torch.tensor(labels, dtype=torch.long)
        return images_pre, labels

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    text_tokens = tokenizer(class_names).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct_top1 = 0
    total = 0

    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    start = time.time()
    for images_pre, labels in loader:
        images_pre = images_pre.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images_pre)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        preds = logits.argmax(dim=-1)

        correct = (preds == labels)
        correct_top1 += correct.sum().item()
        total += labels.size(0)

        for i in range(labels.size(0)):
            cls = labels[i].item()
            class_total[cls] += 1
            if correct[i]:
                class_correct[cls] += 1

    acc = correct_top1 / total
    elapsed = time.time() - start
    print(f"Top-1 accuracy on Food-101 test: {acc * 100:.2f}% ({correct_top1}/{total})")
    print(f"Evaluation time: {elapsed/60:.1f} minutes")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "per_class_accuracy.csv"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("class,correct,total,accuracy\n")
        for idx, name in enumerate(class_names):
            c = class_correct[idx].item()
            t = class_total[idx].item()
            a = c / t if t > 0 else 0.0
            f.write(f"{name},{c},{t},{a:.6f}\n")

    print(f"Saved per-class accuracy to {csv_path}")


if __name__ == "__main__":
    eval_clip_food101()
