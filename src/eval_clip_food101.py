import time

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101

from clip_food101_baseline import load_clip_model


@torch.no_grad()
def eval_clip_food101(batch_size: int = 64, root: str = "../data"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess, tokenizer = load_clip_model()
    model.to(device)
    model.eval()

    dataset = Food101(root=root, split="test")
    class_names = dataset.classes
    print(f"Loaded Food-101 test split with {len(dataset)} images and {len(class_names)} classes")

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

    start = time.time()
    for images_pre, labels in loader:
        images_pre = images_pre.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images_pre)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        preds = logits.argmax(dim=-1)

        correct_top1 += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct_top1 / total
    elapsed = time.time() - start
    print(f"Top-1 accuracy on Food-101 test: {acc * 100:.2f}% ({correct_top1}/{total})")
    print(f"Evaluation time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    eval_clip_food101()
