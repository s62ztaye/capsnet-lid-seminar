# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, caps_dim=8, kernel_size=9, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 256, kernel_size, stride)
        self.caps_dim = caps_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), 32, self.caps_dim, -1)
        x = x.permute(0, 1, 3, 2).contiguous()  # (batch, 32, seq_len, dim)
        return x

class LanguageCaps(nn.Module):
    def __init__(self, num_capsules=50, input_dim=8, out_dim=16, num_routes=32*61, routing_iters=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters
        self.W = nn.Parameter(torch.randn(num_capsules, num_routes, input_dim, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.caps_dim)  # (b, num_routes, dim)
        u_hat = torch.matmul(x.unsqueeze(2), self.W)  # (b, num_routes, num_caps, dim_out)
        u_hat = u_hat.squeeze(3).permute(0, 2, 1, 3)  # (b, num_caps, num_routes, dim_out)

        b = torch.zeros(batch_size, self.num_capsules, x.size(1)).to(x.device)
        for i in range(self.routing_iters):
            c = F.softmax(b, dim=1)  # (b, num_caps, num_routes)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2)  # weighted sum
            v = self.squash(s)
            if i < self.routing_iters - 1:
                b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1)
        return v

    @staticmethod
    def squash(x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2)
        return scale * x / (norm + 1e-8)

class CapsNetLID(nn.Module):
    def __init__(self, vocab_size=95, embed_dim=64, num_languages=50, seq_len=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.primary_caps = PrimaryCaps(embed_dim)
        self.digit_caps = LanguageCaps(num_languages, 8, 16, num_routes=32*((seq_len-8)//2-8)//2)
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_languages, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, seq_len * vocab_size),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = self.embedding(x).transpose(1, 2)  # (b, embed, seq)
        x = F.relu(self.primary_caps.conv(x))  # (b, 256, ~61)
        x = self.primary_caps(x)  # (b, 32, ~61, 8)
        x = self.digit_caps(x)  # (b, 50, 16)
        lengths = torch.norm(x, dim=-1)  # (b, 50)

        # Reconstruction
        if y is None:
            _, max_idx = lengths.max(dim=1)
            y = torch.eye(self.digit_caps.num_capsules, device=x.device)[max_idx]
        else:
            y = torch.eye(self.digit_caps.num_capsules, device=x.device)[y]
        v_masked = (x * y.unsqueeze(-1)).view(x.size(0), -1)
        recon = self.decoder(v_masked).view(x.size(0), -1, vocab_size)
        return lengths, recon
