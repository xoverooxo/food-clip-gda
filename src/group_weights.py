# Group-specific alpha weights for CLIP+GDA fusion
# alpha = weight on text features (1-alpha = weight on image/GDA features)
#
# FINAL OPTIMAL CONFIG based on experiments:
#   - α=0.30 for Group A: +1.44% improvement (best tested)
#   - α=0.50 for all others: baseline is optimal
#
# Experiment history:
#   Group A: α=0.15 → +1.36%, α=0.30 → +1.44% (winner)
#   Group B: α=0.40 → -2.03%, α=0.45 → -0.89% (keep baseline)

group_weights = {
    "A": 0.30,   # Hardest 5 classes - 70% GDA, 30% text
    "B": 0.50,   # Hard classes - baseline is best
    "C": 0.50,   # Easy classes - baseline
    "Other": 0.50,  # Medium difficulty - baseline
}