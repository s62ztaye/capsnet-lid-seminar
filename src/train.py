# train.py
import torch
from torch.utils.data import DataLoader
from model import CapsNetLID
from preprocess import LIDataset
import torch.optim as optim
import torch.nn.functional as F

def margin_loss(predictions, labels, m_plus=0.9, m_minus=0.1, lambda_=0.5):
    T = torch.eye(50)[labels].to(predictions.device)
    L = T * F.relu(m_plus - predictions)**2 + lambda_ * (1 - T) * F.relu(predictions - m_minus)**2
    return L.sum(dim=1).mean()

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset = LIDataset('wili-2018.txt', lang_subset=50)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CapsNetLID(vocab_size=95, num_languages=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            caps_out, recon = model(texts, labels)

            # Margin loss
            loss = margin_loss(caps_out, labels)

            # Reconstruction loss
            targets = F.one_hot(texts, num_classes=95).float().to(device)
            recon_loss = F.mse_loss(recon, targets)
            loss += 0.0005 * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/20 | Loss: {total_loss/len(loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "capsnet_lid.pth")
    print("Model saved as capsnet_lid.pth")

if __name__ == "__main__":
    train()
