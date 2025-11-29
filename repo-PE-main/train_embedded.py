import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from embedding import Encoder, EncoderWrapper, VICRegLoss
from embedding_dataset import HLSpectrogramDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


# setup
GPU = 3
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("CUDA version:", torch.version.cuda)
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
data_dir = '/home/stevenjames.henderson/data/spectrograms/'
save_dir = '/home/stevenjames.henderson/data/models/'
os.makedirs(save_dir, exist_ok=True)

# Data

# Collate function (ChatGPT)
def spectrogram_collate_fn(batch):
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        raise ValueError("All samples in batch were None or invalid!")

    try:
        J = len(batch[0])
        for i, sample in enumerate(batch):
            if len(sample) != J:
                raise ValueError(f"Sample {i} has {len(sample)} items, expected {J}.")

        flat = [x for sample in batch for x in sample]  # flatten to [B*J, C, F, T]
        stacked = torch.stack(flat)
        return stacked.view(len(batch), J, *stacked.shape[1:])

    except Exception as e:
        print("Collate function failed:")
        for i, sample in enumerate(batch):
            if sample is not None:
                print(f"Sample {i} shape summary: {[x.shape for x in sample]}")
            else:
                print(f"Sample {i} is None")
        raise e

dataset = HLSpectrogramDataset(data_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, collate_fn=spectrogram_collate_fn, pin_memory=True, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=spectrogram_collate_fn, pin_memory=True, persistent_workers=True)

# Plot for montioring (Chatgpt)
def plot_tsne(emb1, emb2, epoch):
    """
    Visualizes emb1 and emb2 in 2D using t-SNE.
    """
    emb1 = emb1.detach().cpu().numpy().reshape(-1, emb1.shape[-1])
    emb2 = emb2.detach().cpu().numpy().reshape(-1, emb2.shape[-1])
    
    tsne = TSNE(n_components=2, perplexity=10, random_state=0)
    embeddings = np.concatenate([emb1, emb2], axis=0)
    tsne_result = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(tsne_result[:len(emb1), 0], tsne_result[:len(emb1), 1], label="Channel 1", alpha=0.7)
    plt.scatter(tsne_result[len(emb1):, 0], tsne_result[len(emb1):, 1], label="Channel 2", alpha=0.7)
    plt.legend()
    plt.title(f"t-SNE of Embeddings - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"tsne_epoch_conv_{epoch}.png")
    plt.close()

# Model
embedding_dim = 128
encoder1 = Encoder(embedding_dim=embedding_dim)
encoder2 = Encoder(embedding_dim=embedding_dim)
model = EncoderWrapper(encoder1, encoder2)
model = model.to(device)

# Loss and optimzer
criterion = VICRegLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 20
train_loss_history = []
val_loss_history = []
if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    # print(f'total batch size: {len(train_dataloader)}')
    for batch in train_dataloader:
        batch = batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        with autocast():
            emb = model(batch)
            emb1 = emb[:, :, 0, :]
            emb2 = emb[:, :, 1, :]

            if torch.isnan(emb1).any():
                # print("NaN detected in emb1, skipping...")
                nan_locs = torch.isnan(emb1)
                continue
            loss = criterion(emb1, emb2)
        # if torch.isnan(loss):
            # print("!!! NaN loss detected !!!")
            # print("emb1:", emb1[0, 0, :10])
            # print("emb2:", emb2[0, 0, :10])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)

    # Validation phase
    model.eval()
    total_val_lost = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            emb = model(batch)
            emb1 = emb[:, :, 0, :]
            emb2 = emb[:, :, 1, :]
            # print("emb1 mean/std:", emb1.mean().item(), emb1.std().item())
            # print("emb2 mean/std:", emb2.mean().item(), emb2.std().item())
            # print("cosine sim:", F.cosine_similarity(emb1[0], emb2[0], dim=-1).mean().item())

            if torch.isnan(emb1).any():
                # print("NaN detected in emb1, skipping...")
                continue
            loss = criterion(emb1, emb2)
            total_val_lost += loss.item()
    avg_val_loss = total_val_lost / len(val_dataloader)

    if epoch % 5 == 0:
        # Check for NaNs
        if np.isnan(emb1.detach().cpu().numpy()).any() or np.isnan(emb2.detach().cpu().numpy()).any():
            # print("Skipping t-SNE: embeddings contain NaNs.")
            continue  # or raise Exception("NaNs in embeddings")
        plot_tsne(emb1, emb2, epoch)
    torch.cuda.empty_cache()
    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")


# Save the model yay
torch.save(model.state_dict(), os.path.join(save_dir, 'conv_model.pth'))

# === PLOT LOSS CURVE (ChatGPT)===
plt.figure()
plt.plot(train_loss_history, marker='o')
plt.plot(val_loss_history, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.grid(True)
plt.savefig("../../data/conv_loss_curve.png")
plt.show()
