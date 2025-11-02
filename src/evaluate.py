# evaluate.py
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model, loader, device, num_classes=10):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            truths.extend(y.cpu().numpy())

    acc = accuracy_score(truths, preds)
    print(f"Test Accuracy: {acc:.4f}")

    # Confusion Matrix (Top 10)
    cm = confusion_matrix(truths, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm[:num_classes, :num_classes], annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Top 10 Languages)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return acc
