import numpy as np
import torch
import h5py
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import SpectrogramParamDataset
from flows import make_flow

from IPython.display import clear_output
from time import sleep
import corner

num_files = 1

data_dir = '/Users/shreyaggarwal/Desktop/Project8581/data/train/'

print('Loading Data')
torch_dataset = SpectrogramParamDataset(data_dir, 10, weights='emb.pth', progress=True)

train_set_size = 0.8
val_set_size = 0.1
test_set_size = 0.1

train_data, val_data, test_data = torch.utils.data.random_split(
    torch_dataset, [train_set_size, val_set_size, test_set_size])

del torch_dataset

TRAIN_BATCH_SIZE = 30
VAL_BATCH_SIZE = 30

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
print('Data Loaded, num_data =', torch_dataset.shape[0])
del torch_dataset

flow = make_flow()

for idx, val in enumerate(train_data_loader, 1):
    if idx == 1:
        theta, data = val
        print(theta.shape, data.shape)

num_augmentations = 10

train_loss = []
val_loss = []
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for idx, val in enumerate(train_data_loader, 1):
        theta, data = val
        theta = theta[:, 0]
        #print(augmented_theta.shape, augmented_data.shape)
        #augmented_theta = augmented_theta[:,0:2]
        #print(augmented_theta.shape, augmented_data.shape)

        loss = 0

        flow_loss = -flow.log_prob(theta, context=data).mean()
        optimizer.zero_grad()
        optimizer.step()

        loss += flow_loss.item()

        running_loss += loss/num_augmentations
        train_loss.append(running_loss)
        if idx % 10 == 0:
            last_loss = running_loss / 10 # avg loss
            #print(' Avg. train loss/batch after {} batches = {:.4f}'.format(idx, last_loss))
            tb_x = epoch_index * len(train_data_loader) + idx
            tb_writer.add_scalar('Flow Loss/train', last_loss, tb_x)
            tb_writer.flush()
            running_loss = 0.
    return last_loss


def val_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for idx, val in enumerate(val_data_loader, 1):
        theta, data = val
        theta = theta[:, 0]
        #augmented_theta = augmented_theta[:,0:2]

        loss = 0

        flow_loss = -flow.log_prob(theta, context=data).mean()
        loss += flow_loss.item()

        running_loss += loss/num_augmentations
        val_loss.append(running_loss)
        if idx % 5 == 0:
            last_loss = running_loss / 5
            tb_x = epoch_index * len(val_data_loader) + idx + 1
            tb_writer.add_scalar('Flow Loss/val', last_loss, tb_x)

            tb_writer.flush()
            running_loss = 0.
    tb_writer.flush()
    return last_loss

optimizer = optim.Adam(flow.parameters(), lr=1e-3)

scheduler_1 = optim.lr_scheduler.ConstantLR(optimizer, total_iters=5)
scheduler_2 = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=20, max_lr=1e-3)
scheduler_3 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[5, 20])

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("gw-norm.pth", comment="With LR=1e-3", flush_secs=5)
epoch_number = 0

EPOCHS = 100

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    # Gradient tracking
    flow.train(True)
    avg_train_loss = train_one_epoch(epoch_number, writer)

    # no gradient tracking, for validation
    flow.train(False)
    avg_val_loss = val_one_epoch(epoch_number, writer)

    #print(f"Train/Val flow Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")
    #for param_group in optimizer.param_groups:
    #    #print("Current LR = {:.3e}".format(param_group['lr']))
    epoch_number += 1
    try:
        scheduler.step(avg_val_loss)
    except TypeError:
        scheduler.step()

PATH = './gw-norm1.pth'
torch.save(flow.state_dict(), PATH)

PATH = './gw-norm1.pth'
device='cpu'
flow.load_state_dict(torch.load(PATH, map_location=device))
flow.eval()

def live_plot_samples(samples, truth):
    print(truth)
    clear_output(wait=True)
    sleep(0.5)
    print(samples.shape)
    figure = corner.corner(
        samples.numpy(), quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        #labels=list(gwpriors.keys()),
        truth=truth,
    )

    corner.overplot_lines(figure, truth, color="C1")
    corner.overplot_points(figure, truth[None], marker="s", color="C1")

n_samples=256//2

for idx, (theta_test, data_test) in enumerate(test_data_loader):
    if idx % 450 !=0: continue 
    with torch.no_grad():
        samples = flow.sample(3000, context=data_test)
    live_plot_samples(samples[0].cpu(), theta_test[0, 0].cpu())
    plt.show()
