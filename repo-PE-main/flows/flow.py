

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform


def make_flow(input_dim=6, context_dim=128*2, num_transforms=5, num_blocks=4, hidden_features=30):
    transforms = []
    for _ in range(num_transforms):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=input_dim,
                hidden_features=hidden_features,
                context_features=context_dim,
                num_blocks=num_blocks,
                use_residual_blocks=True
            )
        )
        transforms.append(RandomPermutation(features=input_dim))

    transform = CompositeTransform(transforms)
    base_dist = StandardNormal([input_dim])
    flow = Flow(transform, base_dist)
    return flow







# class FlowModel(pl.LightningModule):
#     def __init__(self, input_dim, context_dim, lr=1e-3):
#         super().__init__()
#         self.save_hyperparameters()
#         self.flow = make_flow(input_dim, context_dim)

#     def forward(self, context, parameters):
        
#         return -self.flow.log_prob(inputs=parameters, context=context)

#     def training_step(self, batch, batch_idx):
#         context, parameters = batch
#         loss = self(context, parameters).mean()
#         self.log("train_loss", loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         context, parameters = batch
#         loss = self(context, parameters).mean()
#         self.log("val_loss", loss, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=2e-3)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1)
#         return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}




# # datamodule.py
# import torch
# import pytorch_lightning as pl

# class DummyDataModule(pl.LightningDataModule):   ## change this to actual datamodule
#     def __init__(self, num_samples=10000, batch_size=128, input_dim=8, context_dim=64):
#         super().__init__()
#         self.num_samples = num_samples
#         self.batch_size = batch_size
#         self.input_dim = input_dim
#         self.context_dim = context_dim

#     def setup(self, stage=None):
#         self.context = torch.randn(self.num_samples, self.context_dim)
#         self.parameters = torch.randn(self.num_samples, self.input_dim)

#     def train_dataloader(self):
#         dataset = torch.utils.data.TensorDataset(self.context, self.parameters)
#         return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         dataset = torch.utils.data.TensorDataset(self.context, self.parameters)
#         return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)