import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from flow import Encoder, EncoderWrapper, make_flow
from dataset import SpectrogramParamDataset
from tqdm import tqdm

# ==== Hyperparameters ====
embedding_dim = 128
param_dim = 7
num_epochs = 20
batch_size = 2
lr = 1e-3
num_transforms = 5
num_blocks = 4
hidden_features = 30

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load dataset and split ====
data_path = "./data/spectrograms"
dataset = SpectrogramParamDataset(data_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==== Initialize encoders and flow ====
encoder1 = Encoder(embedding_dim).to(device)
encoder2 = Encoder(embedding_dim).to(device)
embedding_net = EncoderWrapper(encoder1, encoder2).to(device)

context_dim = embedding_dim * 2 * 10  # 10 time shifts, 2 channels
flow = make_flow(input_dim=param_dim, context_dim=context_dim,
                 num_transforms=num_transforms, num_blocks=num_blocks,
                 hidden_features=hidden_features).to(device)

# ==== Optimizer ====
optimizer = optim.Adam(list(flow.parameters()) + list(embedding_net.parameters()), lr=lr)

# ==== Training loop ====
for epoch in range(num_epochs):
    flow.train()
    embedding_net.train()

    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        x, params = batch  # x: [B, J, 2, F, T], params: [B, P]
        x, params = x.to(device), params.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            context = embedding_net(x)  # [B, context_dim]
            loss = -flow.log_prob(inputs=params, context=context).mean()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: loss = {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Avg NLL Loss = {avg_loss:.4f}")

    # ==== Validation ====
    flow.eval()
    embedding_net.eval()

    val_loss = 0
    with torch.no_grad():
        for x_val, params_val in val_loader:
            x_val, params_val = x_val.to(device), params_val.to(device)
            context_val = embedding_net(x_val)
            loss_val = -flow.log_prob(inputs=params_val, context=context_val).mean()
            val_loss += loss_val.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Val NLL Loss = {avg_val_loss:.4f}")

# ==== Save model ====
torch.save({
    'flow_state_dict': flow.state_dict(),
    'encoder1_state_dict': encoder1.state_dict(),
    'encoder2_state_dict': encoder2.state_dict()
}, "flow_model.pth")
