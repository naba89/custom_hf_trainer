"""Copyright: Nabarun Goswami (2023)."""
import os
from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments
from transformers.modeling_outputs import BaseModelOutput

from custom_hf_trainer import CustomTrainer


# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.rand(size, 10)  # Random data
        self.labels = torch.randint(0, 2, (size,))  # Binary labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'input_ids': self.data[idx], 'labels': self.labels[idx]}


@dataclass
class DummyModelOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    dummy_loss: torch.FloatTensor = None


# Dummy Model
class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, input_ids, labels=None) -> DummyModelOutput:
        outputs = self.linear(input_ids)
        loss = None
        # Add a dummy loss for demonstration
        dummy_loss = torch.tensor(0.5).float()  # This can be any computation you define
        loss = ((outputs - outputs.detach()) + 0.7).mean()
        return DummyModelOutput(loss=loss, logits=outputs, dummy_loss=dummy_loss)

    def get_extra_loss_indices(self):
        loss_index_mapping = {
            "dummy_loss": -1,
        }
        return loss_index_mapping


def compute_metrics(eval_pred, loss_index_mapping=None):
    model_output = eval_pred.predictions
    out_dict = {}
    if loss_index_mapping is not None:
        for k, v in loss_index_mapping.items():
            out_dict[k] = model_output[v].mean().item()
    return out_dict


os.environ["WANDB_PROJECT"] = "dummy_project"

# Create dataset and model instances
dataset = DummyDataset(size=8000)
model = DummyModel()

persistent_workers = False  # set to True to enable persistent workers

# Training arguments
training_args = TrainingArguments(
    output_dir="./test_trainer",
    run_name=f'dataloader_peristent_workers={persistent_workers}',
    num_train_epochs=200,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    dataloader_num_workers=8,
    dataloader_persistent_workers=persistent_workers,
    logging_strategy="no",
    eval_strategy="epoch",
)

# get the extra loss indices from the model
extra_loss_index_mapping = model.get_extra_loss_indices()

# Initialize the custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    extra_losses=list(extra_loss_index_mapping.keys()),
    compute_metrics=partial(compute_metrics, loss_index_mapping=extra_loss_index_mapping)
)

# Train the model
trainer.train()
