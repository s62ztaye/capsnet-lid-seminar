# visualize.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_routing_example():
    # Simulated routing weights for "example"
    word = "example"
    languages = ["English", "French", "German", "Spanish", "Italian"]
    weights = [
        [0.8, 0.1, 0.05, 0.02, 0.03],
        [0.7, 0.05, 0.08, 0.12, 0.05],
        [0.6, 0.07, 0.1, 0.15, 0.08],
        [0.75, 0.05, 0.05, 0.1, 0.05],
        [0.8, 0.03, 0.07, 0.06, 0.04],
        [0.82, 0.02, 0.06, 0.05, 0.05],
        [0.85, 0.01, 0.04, 0.05, 0.05],
    ]

    plt.figure(figsize=(10, 4))
    sns.heatmap(weights, annot=True, xticklabels=languages, yticklabels=list(word), cmap="YlOrRd")
    plt.title("Routing Weights: Character Position → Language Capsule")
    plt.ylabel("Character")
    plt.xlabel("Language Capsules")
    plt.tight_layout()
    plt.show()
