from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Food101

from clip_food101_baseline import load_clip_model


@torch.no_grad()
def run_demo(num_images: int = 8, root: str = "../data"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, preprocess, tokenizer = load_clip_model()
    model.to(device)
    model.eval()

    dataset = Food101(root=root, split="test")
    class_names = dataset.classes
    print(f"Loaded Food-101 with {len(dataset)} images and {len(class_names)} classes")

    indices = list(range(min(num_images, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)

    def collate_fn(batch):
        images, labels = zip(*batch)  # list of PIL images, list of ints
        images_pre = torch.stack([preprocess(img) for img in images])
        labels = torch.tensor(labels, dtype=torch.long)
        return images_pre, labels

    loader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    text_tokens = tokenizer(class_names).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for batch_idx, (images_pre, labels) in enumerate(loader):
        images_pre = images_pre.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images_pre)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        top_probs, top_idxs = logits.softmax(dim=-1).topk(1, dim=-1)

        for i in range(images_pre.size(0)):
            idx = batch_idx * loader.batch_size + i
            true_class = class_names[labels[i].item()]
            pred_class = class_names[top_idxs[i, 0].item()]
            prob = top_probs[i, 0].item()
            print(
                f"Image {idx:3d}: true={true_class:20s}  "
                f"pred={pred_class:20s}  p={prob:.3f}"
            )


if __name__ == "__main__":
    run_demo()
