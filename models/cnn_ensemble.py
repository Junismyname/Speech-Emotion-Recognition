import torch
import torch.nn as nn

class EnsembleBaseCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EnsembleBaseCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=8),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        dummy = torch.zeros(1, 1, input_dim)
        flatten_size = self.features(dummy).numel()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, num_classes)
        )

    def forward(self, x):
        # Permute (Batch, Length, Channels) to PyTorch's expected (Batch, Channels, Length)
        if x.shape[-1] == 1:
            x = x.permute(0, 2, 1)

        x = self.features(x)
        return self.classifier(x)