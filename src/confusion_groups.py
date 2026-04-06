# Difficulty groups based on zero-shot CLIP per-class accuracy
# Group A: Hardest classes (< 50% accuracy) - need most help
# Group B: Hard classes (50-70% accuracy) - struggling
# Group C: Easy classes (> 90% accuracy) - already doing well
# Other: Medium difficulty (70-90% accuracy)

difficulty_groups = {
    # Hardest classes - severe failures (< 50%)
    "A": [
        "waffles",           # 0.4%
        "pork_chop",         # 24.8%
        "gnocchi",           # 44.0%
        "cannoli",           # 46.0%
        "ravioli",           # 47.6%
    ],
    
    # Hard classes - struggling (50-70%)
    "B": [
        "omelette",                  # 51.6%
        "steak",                     # 52.4%
        "apple_pie",                 # 57.6%
        "strawberry_shortcake",      # 59.2%
        "lasagna",                   # 60.8%
        "foie_gras",                 # 61.2%
        "breakfast_burrito",         # 62.4%
        "grilled_cheese_sandwich",   # 64.4%
        "pancakes",                  # 64.4%
        "cheesecake",                # 65.2%
        "chocolate_mousse",          # 67.6%
        "risotto",                   # 68.4%
        "ice_cream",                 # 68.8%
        "tuna_tartare",              # 69.2%
    ],
    
    # Easy classes - already performing well (> 90%)
    "C": [
        "edamame",               # 99.6%
        "oysters",               # 98.8%
        "pad_thai",              # 97.6%
        "club_sandwich",         # 97.2%
        "pho",                   # 96.4%
        "lobster_roll_sandwich", # 96.0%
        "macarons",              # 96.0%
        "spaghetti_bolognese",   # 95.6%
        "bibimbap",              # 95.2%
        "hot_and_sour_soup",     # 94.8%
        "frozen_yogurt",         # 94.4%
        "cheese_plate",          # 93.6%
        "fish_and_chips",        # 93.6%
        "beignets",              # 93.2%
        "creme_brulee",          # 93.2%
        "fried_calamari",        # 93.2%
        "huevos_rancheros",      # 93.2%
        "spaghetti_carbonara",   # 93.2%
        "lobster_bisque",        # 92.8%
        "deviled_eggs",          # 92.4%
        "eggs_benedict",         # 92.4%
        "seaweed_salad",         # 92.4%
        "caesar_salad",          # 92.0%
        "dumplings",             # 92.0%
        "mussels",               # 92.0%
        "takoyaki",              # 91.2%
        "caprese_salad",         # 90.4%
        "chicken_wings",         # 90.8%
        "pizza",                 # 90.8%
        "spring_rolls",          # 90.8%
        "sushi",                 # 90.8%
        "beef_carpaccio",        # 90.0%
        "paella",                # 90.0%
    ],
}