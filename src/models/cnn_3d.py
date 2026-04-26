import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    """ResNet-style basic block for 3D features (keeps spatial size)."""
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(channels)
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)
        return out

class Basic3DCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.2, res_drop: float = 0.0):
        super().__init__()

        def down_block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2),  # downsample x2
            )

        self.features = nn.Sequential(
            down_block(in_channels, 16),
            ResBlock3D(16, dropout=res_drop),

            down_block(16, 32),
            ResBlock3D(32, dropout=res_drop),

            down_block(32, 64),
            ResBlock3D(64, dropout=res_drop),

            down_block(64, 128),
            ResBlock3D(128, dropout=res_drop),
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
