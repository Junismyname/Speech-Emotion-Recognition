import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super(CNN1D, self).__init__()
        
        # Block 1: Conv1D -> ReLU -> MaxPool1D
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        )
        
        # Block 2: Conv1D -> ReLU -> MaxPool1D
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        )
        
        # Block 3: Conv1D -> ReLU -> MaxPool1D -> Dropout
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2),
            nn.Dropout(0.2)
        )
        
        # Block 4: Conv1D -> ReLU -> MaxPool1D
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        )
        
        # Dynamically calculate the flattened feature size after 4 max-pooling layers
        # Each MaxPool1d with kernel_size=5, stride=2, padding=2 halves the sequence length roughly: L -> L/2 -> L/4 -> L/8 -> L/16
        dummy_input = torch.zeros(1, 1, input_dim)
        with torch.no_grad():
            x = self.block1(dummy_input)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            flatten_size = x.numel()
            
        # Fully Connected Classifier Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        if x.shape[-1] == 1:
            x = x.permute(0, 2, 1)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc_layers(x)
        return x
