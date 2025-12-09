def get_latent_representation(model, cell_data):
    with torch.no_grad():
        if not isinstance(cell_data, torch.Tensor):
            cell_data = torch.DoubleTensor(cell_data)
        
        if len(cell_data.shape) == 1:
            cell_data = cell_data.unsqueeze(0)
        
        mu, _ = model.encode(cell_data)
        
        return mu.cpu().numpy()