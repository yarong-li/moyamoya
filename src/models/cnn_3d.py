import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic3DCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2),  # downsample x2
            )

        self.features = nn.Sequential(
            block(in_channels, 16),   # -> ~ (D/2,H/2,W/2)
            block(16, 32),            # -> /4
            block(32, 64),            # -> /8
            block(64, 128),           # -> /16
        )

        self.pool = nn.AdaptiveAvgPool3d(1)  # -> [B, C, 1,1,1]
        self.classifier = nn.Sequential(
            nn.Flatten(),             # -> [B, C]
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x  # logits
