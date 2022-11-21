#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
from sklearn.metrics import classification_report, confusion_matrix

pl.seed_everything(24)


def load_data(data_dir: Path = "."):
    x_train, y_train = pd.read_csv(data_dir / "X_train.csv"), pd.read_csv(data_dir / "y_train.csv")
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y_train.surface)
    y_train["label"] = encoded_labels

    feature_columns = x_train.columns.values()[3:]
    sequences = [
        (group[feature_columns], y_train[y_train.series_id == series_id].iloc[0].label)
        for series_id, group in x_train.groupby("series_id")
    ]

    train_sequences, test_sequences = train_test_split(sequences, test_size=0.2)

    return train_sequences, test_sequences, label_encoder


class SurfaceDataset(Dataset):
    def __init__(self, ssequences) -> None:
        super().__init__()
        self.sequences = ssequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label=torch.tensor(label).long()
        )


class SurfaceDataModule(pl.LightningDataModule):
    def __init__(self, train_seqs, test_seqs, batch_size: int) -> None:
        super().__init__()
        self.train_seqs = train_seqs
        self.test_seqs = test_seqs
        self.batch_size = batch_size
    
    def setup(self, stage = None) -> None:
        self.train_dataset = SurfaceDataset(self.train_seqs)
        self.test_dataset = SurfaceDataset(self.test_seqs)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count()
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count()
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count()
        )


class SurfaceModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256, n_layers=1) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True
            # dropout=0.75
        )
        self.classifier = nn.Linear(n_hidden, n_classes)
    
    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        out = self.classifier(hidden[-1])
        return out


class SurfacePredictor(pl.LightningDataModule):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.model = SurfaceModel(n_features=n_features, n_classes=n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()
    
    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequences, labels)
        predictions = torch.argmax(output, dim=1)
        step_accuracy = self.metric(predictions, labels)

        self.log("training_loss", loss, prog_bar=True, logger=True)
        self.log("training_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "accuracy": step_accuracy
        }
    
    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequences, labels)
        predictions = torch.argmax(output, dim=1)
        step_accuracy = self.metric(predictions, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "accuracy": step_accuracy
        }
    
    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequences, labels)
        predictions = torch.argmax(output, dim=1)
        step_accuracy = self.metric(predictions, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "accuracy": step_accuracy
        }
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == "__main__":
    n_epochs = 250
    batch_size = 64

    trainset, testset, label_enc = load_data()

    data_module = SurfaceDataModule(
        trainset, testset, batch_size
    )

    model = SurfacePredictor(
        n_features=trainset.shape[1],
        n_classes=len(label_enc.classes_)
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="chkpts",
        filename="best_chkpt",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="surface")

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_cb,
        max_epochs=n_epochs,
        # gpus=1,
        enable_progress_bar=True
    )

    trainer.fit(model, data_module)
