import os
import torch
from PIL import Image

# For loading pre-train model
import pytorch_lightning as pl
from transformers import ViTForImageClassification
from transformers.optimization import AdamW
from torch.utils.data import random_split

# For dataloading
from math import floor
from torch import manual_seed
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

# For loggin loss, acc, ckpt
from pytorch_lightning.loggers import TensorBoardLogger

# Training Data transformation
pokemon_train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class PokemonClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Pretrain Model
        model_name = "google/vit-base-patch16-224"
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.num_labels = 150  # number of pokemon classes
        self.model.classifier = torch.nn.Linear(768, self.model.num_labels)
        # Hyper-parameters
        self.batch_size = 16
        self.lr = 5e-5
        # Data roots
        self.data_root = "C:/study/ee576/project/data/PokemonData"
        self.ckpt_dir = "C:/study/ee576/project/ee576-cv-vit/models/logs/pokemon/version_4/checkpoints"

    def prepare_data(self):
        manual_seed(80)
        dataset = ImageFolder(self.data_root, pokemon_train_transforms)
        val_size = floor((len(dataset)) * 0.10)  # train:val -> 90:10
        train_size = len(dataset) - val_size
        self.train_ds, self.test_ds = random_split(dataset, [train_size, val_size])

    def configure_optimizers(self):  # used by pytorch-lightning
        return AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

    def forward(self, batch, batch_idx):
        return self.model(batch[0].squeeze(), labels=batch[1].squeeze())

    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log("train_loss: ", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch, batch_idx)  # loss, pred_id
        loss = output[0]
        pred_labels = output[1].argmax(-1)
        gt_labels = batch[1].squeeze()
        accuracy = torch.sum(pred_labels == gt_labels) / self.batch_size
        self.log("val_accuracy: ", accuracy)
        self.log("val_loss: ", loss)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, self.batch_size, shuffle=True, num_workers=2, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, num_workers=2, pin_memory=True)


### Fine-Tune(Train) Pokemon Classifier ###
if __name__ == "__main__":
    PokeModel = PokemonClassifier()
    logger = TensorBoardLogger("logs", name="pokemon")
    trainer = pl.Trainer(
        accumulate_grad_batches=4,
        default_root_dir="logs",
        accelerator="gpu",
        devices="auto",
        max_epochs=5,
        check_val_every_n_epoch=1,
        logger=logger,
    )
    trainer.fit(PokeModel)
