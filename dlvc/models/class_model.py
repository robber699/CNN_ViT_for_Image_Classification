import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    
    def save(self, save_dir: str, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        # Define the file path for saving the model
        save_path = Path(save_dir)
        if suffix:
            filename = f"model_{suffix}.pth"
        else:
            filename = "model.pth"
        filepath = save_path / filename
        
        # Save the state_dict of the model
        torch.save(self.net.state_dict(), filepath)
        print(f"Model saved at: {filepath}")

    def load(self, path: str):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        # Load the state_dict of the model
        self.net.load_state_dict(torch.load(path))
        print(f"Model loaded from: {path}")
