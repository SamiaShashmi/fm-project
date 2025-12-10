import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
def interpolate(z1_mean, z2_mean, steps, model=None):        
   
    interpolated_z = []
    for alpha in torch.linspace(0, 1, steps):
        z = alpha * z1_mean + (1 - alpha) * z2_mean
        interpolated_z.append(z)

    if model == None:
        return interpolated_z
    interpolated_outputs = []
    with torch.no_grad():
        for z in interpolated_z:
            output = model.decode(z)[0]
            interpolated_outputs.append(output)
    
    interpolated_outputs = [tensor.numpy().squeeze() for tensor in interpolated_outputs]
    interpolated_outputs = np.stack(interpolated_outputs, axis=0)
    return interpolated_outputs
    
def get_latent_representation(model, cell_data):
    with torch.no_grad():
        if not isinstance(cell_data, torch.Tensor):
            cell_data = torch.DoubleTensor(cell_data)
        
        if len(cell_data.shape) == 1:
            cell_data = cell_data.unsqueeze(0)
        
        mu, _ = model.encode(cell_data)
        
        return mu.cpu().numpy()

class LatentToExprDataset(Dataset):
    def __init__(self, z, x):
        self.z = torch.from_numpy(z)
        self.x = torch.from_numpy(x)

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return self.z[idx], self.x[idx]

class LatentToGeneMLP(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=1024, n_genes=2000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, z):
        return self.net(z)