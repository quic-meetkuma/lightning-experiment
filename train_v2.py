# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
from datetime import datetime

import torch
import torch_qaic
from backend import QAICAccelerator
from lightning.pytorch.strategies import SingleDeviceStrategy
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

import lightning as L

import train_config as tr_config
from dataset.imdb import IMDBDatasetHelper
from models.model_factory import ModelRegistry


class SeqClassifier(L.LightningModule):
    def __init__(self, model, warmup_epochs, steps_per_epoch):
        super().__init__()
        self.model = model
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        new_batch = {}
        for k, v in batch:
            if k != "label":
                new_batch[k] = v
        logits = self.model(**new_batch)  # [B, num_classes]
        return logits

    def training_step(self, batch, idx):
        label = batch.pop("label")  # [B, num_classes]
        outputs = self.model(**batch)  # [B, num_classes]
        logits = outputs.logits
        probs = F.softmax(logits)  # [B, num_classes]
        loss = self.loss_fn(probs, label.long())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        label = batch.pop("label")  # [B, num_classes]
        outputs = self.model(**batch)  # [B, num_classes]
        logits = outputs.logits
        probs = F.softmax(logits)  # [B, num_classes]
        loss = self.loss_fn(probs, label.long())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        num_train_steps = self.steps_per_epoch * self.trainer.max_epochs
        num_warmup_steps = self.steps_per_epoch * self.warmup_epochs

        scheduler = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps, num_train_steps
        )

        return [opt], [{"scheduler": scheduler, "interval": "step"}]


model_class = ModelRegistry.fetch_model("albert")()
model = model_class.get_model()
tokenizer = model_class.get_tokenizer()

train_dataset = IMDBDatasetHelper(
    tokenizer,
    "train",
    tr_config.batch_size,
    tr_config.max_seq_len,
    tr_config.num_workers,
).get_dataloader()
test_dataset = IMDBDatasetHelper(
    tokenizer,
    "test",
    tr_config.batch_size,
    tr_config.max_seq_len,
    tr_config.num_workers,
).get_dataloader()

model.train()

pl_module = SeqClassifier(
    model, tr_config.warmup_epochs, steps_per_epoch=len(train_dataset)
)

now = datetime.now()
formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
exp_path = os.path.join(tr_config.train_data_path, formatted_date_time)

os.makedirs(tr_config.train_data_path, exist_ok=True)
os.makedirs(exp_path)


trainer = L.Trainer(
    accelerator="qaic",
    devices=2,
    strategy="ddp",
    max_epochs=tr_config.num_epochs,
    # strategy=SingleDeviceStrategy(device="qaic"),
    # strategy=DDPStrategy(device="qaic"),
)

trainer.fit(pl_module, train_dataset, test_dataset)
