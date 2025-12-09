import torch
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

    return interpolated_outputs