from pathlib import Path

from torchvision.datasets import Food101
from torchvision import transforms


def download_food101(root: str = "../data"):
    """
    Download the Food-101 dataset using torchvision.

    Files will be placed under:
        <root>/food-101/
    """
    root_path = Path(root).resolve()
    print(f"Downloading Food-101 into: {root_path}")

    transform = transforms.ToTensor()

    print("Downloading training split (this may take several minutes)...")
    Food101(root=str(root_path), split="train", transform=transform, download=True)

    print("Downloading test split (if not already present)...")
    Food101(root=str(root_path), split="test", transform=transform, download=True)

    print("Food-101 download complete.")


if __name__ == "__main__":
    download_food101()
