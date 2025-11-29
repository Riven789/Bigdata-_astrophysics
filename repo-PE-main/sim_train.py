import numpy as np
import torch
import h5py
import os

from dataset import Spectrogram, SpectrogramParamDataset
from embedding import Encoder, EncoderWrapper, VICRegLoss

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import tqdm
import matplotlib.pyplot as plt

num_files = 1

data_dir = '/home/shrey.aggarwal/data/train/'

torch_dataset = SpectrogramParamDataset(data_dir, 3)
print('loaded')
train_set_size = 0.8
val_set_size = 0.1
test_set_size = 0.1

train_data, val_data, test_data = torch.utils.data.random_split(
    torch_dataset, [train_set_size, val_set_size, test_set_size])

TRAIN_BATCH_SIZE = 175
VAL_BATCH_SIZE = 80

train_data_loader = DataLoader(
    train_data, batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)

val_data_loader = DataLoader(
    val_data, batch_size=VAL_BATCH_SIZE,
    shuffle=True
)

test_data_loader = DataLoader(
    test_data, batch_size=1,
    shuffle=False
)

num_epochs = 1
embedding_dim = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
encoder1 = Encoder(embedding_dim).to(device)
encoder2 = Encoder(embedding_dim).to(device)

criterion = VICRegLoss()
optimizer = optim.Adam(list(encoder1.parameters()) + list(encoder2.parameters()), lr=1e-3)

train_losses = []
val_losses = []


i = 0
t = 10

len_train = len(train_data_loader)
wrapper = EncoderWrapper(encoder1, encoder2).to(device)

for epoch in range(num_epochs):

    print(epoch)
    encoder1.train()
    encoder2.train()
    total_train_loss = 0
    total_val_loss = 0

    # --- Training ---
    for param_data, batch_data in train_data_loader:
        batch_data = batch_data.to(device)
        param_data = param_data.to(device)  # optional: not used in VICReg

        emb1, emb2 = wrapper(batch_data)

        # VICReg loss across time shifts
        loss1 = 0
        loss2 = 0
        for i in range(t-1):
            loss1 += criterion(emb1[:, i, :], emb1[:, i+1, :])[0]
            loss2 += criterion(emb2[:, i, :], emb2[:, i+1, :])[0]
            print(epoch, i, 'train', loss1, loss2)
        loss = (loss1 + loss2) / (2 * (t-1))
        print(loss)
        total_train_loss += loss.item()
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # --- Validation ---
    encoder1.eval()
    encoder2.eval()
    
    with torch.no_grad():
        for param_data, batch_data in val_data_loader:
            batch_data = batch_data.to(device)
            param_data = param_data.to(device)  # optional: not used in VICReg

            emb1, emb2 = wrapper(batch_data)

            loss1 = 0
            loss2 = 0
            for i in range(t-1):
                loss1 += criterion(emb1[:, i, :], emb1[:, i+1, :])[0]
                loss2 += criterion(emb2[:, i, :], emb2[:, i+1, :])[0]
                print(epoch, i, 'val', loss1, loss2)
            loss = (loss1 + loss2) / (2 * (t-1))
            total_val_loss += loss.item()
    print(total_train_loss, len(train_data_loader))
    avg_train_loss = total_train_loss / TRAIN_BATCH_SIZE
    avg_val_loss = total_val_loss / VAL_BATCH_SIZE

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig('sine_wave.png')
plt.show()

# Save the final pretrained encoders
torch.save({
    'encoder1_state_dict': encoder1.state_dict(),
    'encoder2_state_dict': encoder2.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f='emb.pth')


