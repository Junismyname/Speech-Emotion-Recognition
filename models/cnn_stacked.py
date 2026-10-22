import torch
import torch.nn as nn

class CNNStacked(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super(CNNStacked, self).__init__()
        
        # Block 1: Double Conv1D -> Dropout -> Massive MaxPool
        # padding=2 acts as Keras padding='same' for kernel_size=5
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=8, stride=8) # Massive downsampling
        )
        
        # Block 2: Double Conv1D
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        
        # Dynamically calculate the flattened feature size
        dummy_input = torch.zeros(1, 1, input_dim)
        with torch.no_grad():
            x = self.block1(dummy_input)
            x = self.block2(x)
            flatten_size = x.numel()
            
        # Direct Classification Head (No Hidden Dense Layer)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, num_classes)
        )

    def forward(self, x):
        # Permute (Batch, Length, Channels) to PyTorch's expected (Batch, Channels, Length)
        if x.shape[-1] == 1:
            x = x.permute(0, 2, 1)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x