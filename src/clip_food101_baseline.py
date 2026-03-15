import torch
import open_clip


def load_clip_model(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    """Load a CLIP model and its preprocessing."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


if __name__ == "__main__":
    model, preprocess, tokenizer = load_clip_model()
    print("Loaded CLIP model:", type(model))
