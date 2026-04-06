# compute_group_weights.py
# Compute a simple per-group weight alpha_g from the confusion matrix.
# Higher internal confusion -> lower alpha_g (less trust in image/GDA).

from pathlib import Path
import csv

from confusion_groups import get_group_for_class


def load_confusion_matrix(csv_path: Path):
    # Returns: class_names (list), matrix (list of list of ints)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # header[0] is empty cell, then class names
        class_names = header[1:]

        matrix = []
        row_names = []
        for row in reader:
            row_names.append(row[0])
            values = [int(x) for x in row[1:]]
            matrix.append(values)

    # Sanity check: header class order should match row order
    if class_names != row_names:
        print("WARNING: header class names and row names differ.")
    return class_names, matrix


def compute_group_confusion(class_names, matrix):
    """
    For each group g, compute:
      - total_true_g: total examples whose true class is in group g
      - internal_mistakes_g: off-diagonal predictions within the same group
      - confusion_ratio_g = internal_mistakes_g / total_true_g
    """
    n = len(class_names)

    # Map index -> class name and group
    idx_to_class = {i: cls for i, cls in enumerate(class_names)}
    idx_to_group = {i: get_group_for_class(cls) for i, cls in enumerate(class_names)}

    stats = {}  # group -> dict
    for g in set(idx_to_group.values()):
        stats[g] = {
            "total_true": 0,
            "internal_mistakes": 0,
        }

    for i in range(n):
        g_i = idx_to_group[i]
        row = matrix[i]
        total_true_i = sum(row)
        stats[g_i]["total_true"] += total_true_i

        # internal mistakes: j != i, but same group
        for j in range(n):
            if j == i:
                continue
            if idx_to_group[j] == g_i:
                stats[g_i]["internal_mistakes"] += row[j]

    # Compute confusion ratio per group
    for g, s in stats.items():
        if s["total_true"] > 0:
            s["confusion_ratio"] = s["internal_mistakes"] / s["total_true"]
        else:
            s["confusion_ratio"] = 0.0

    return stats


def confusion_ratio_to_alpha(confusion_ratio, alpha_min=0.2, alpha_max=0.8):
    """
    Map confusion_ratio in [0,1] to a weight alpha in [alpha_min, alpha_max].
    More confusion -> smaller alpha (less trust in GDA).
    We clip ratios to [0, 0.5] for stability.
    """
    r = max(0.0, min(confusion_ratio, 0.5))
    # Linear mapping: r=0 -> alpha_max, r=0.5 -> alpha_min
    if 0.5 == 0:
        return alpha_max
    t = r / 0.5
    return alpha_max * (1 - t) + alpha_min * t


def main():
    # Path to your confusion matrix file
    csv_path = Path("../results/clip_food101_ViT-B_32_test/confusion_matrix.csv")

    class_names, matrix = load_confusion_matrix(csv_path)
    stats = compute_group_confusion(class_names, matrix)

    print("Group confusion stats and suggested alphas:")
    for g, s in stats.items():
        cr = s["confusion_ratio"]
        alpha = confusion_ratio_to_alpha(cr)
        print(
            f"{g:20s} total_true={s['total_true']:6d} "
            f"internal_mistakes={s['internal_mistakes']:6d} "
            f"confusion_ratio={cr:.4f}  alpha={alpha:.3f}"
        )


if __name__ == "__main__":
    main()