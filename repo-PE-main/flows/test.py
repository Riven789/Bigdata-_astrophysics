import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import corner
from torch.utils.data import DataLoader
from flow import Encoder, EncoderWrapper, make_flow
from dataset import SpectrogramParamDataset

# ==== Settings ====
data_path = "./data/spectrograms"
batch_size = 1
param_dim = 7
embedding_dim = 128
context_dim = embedding_dim * 2 * 10  # 10 time shifts, 2 channels
num_transforms = 5
num_blocks = 4
hidden_features = 30
n_samples = 1000

save_dir = "test_outputs"
os.makedirs(save_dir, exist_ok=True)

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load model ====
encoder1 = Encoder(embedding_dim).to(device)
encoder2 = Encoder(embedding_dim).to(device)
embedding_net = EncoderWrapper(encoder1, encoder2).to(device)

flow = make_flow(param_dim, context_dim, num_transforms, num_blocks, hidden_features).to(device)

checkpoint = torch.load("flow_model.pth", map_location=device)
flow.load_state_dict(checkpoint['flow_state_dict'])
encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

flow.eval()
embedding_net.eval()

# ==== Load test dataset ====
dataset = SpectrogramParamDataset(data_path)
test_size = int(0.1 * len(dataset))
test_dataset = torch.utils.data.Subset(dataset, range(-test_size, 0))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==== Evaluation ====
param_names = ['chirp_mass', 'mass_ratio', 'chi1', 'chi2', 'distance', 'phic', 'inclination']

for idx, (x, true_params) in enumerate(test_loader):
    if idx >= 10:
        break

    x = x.to(device)
    true_params = true_params.squeeze(0).cpu().numpy()

    with torch.no_grad():
        context = embedding_net(x)
        samples = flow.sample(n_samples, context=context)

    samples_np = samples.squeeze(0).cpu().numpy()  # shape: [n_samples, param_dim]
    np.save(os.path.join(save_dir, f"posterior_samples_{idx}.npy"), samples_np)

    figure = corner.corner(
        samples_np,
        labels=param_names,
        truths=true_params,
        show_titles=True,
        title_fmt=".2f"
    )
    figure.savefig(os.path.join(save_dir, f"corner_plot_{idx}.png"))
    plt.close(figure)

    print(f" Saved posterior and plot for test sample {idx}")
