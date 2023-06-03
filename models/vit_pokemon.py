import torch
import pytorch_lightning as pl
from PIL import Image

# from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import ViTForImageClassification, ViTImageProcessor
import os
import re
from tqdm import tqdm
from transformers.optimization import AdamW
from pytorch_lightning.callbacks import Callback
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torch import manual_seed
from math import floor
from torchvision import transforms

from pytorch_lightning.loggers import TensorBoardLogger

# save_path = os.path.join("C:/study/ee576/project/ee576-cv-vit/ckpt")
# if not os.path.exists(save_path):
#     os.mkdir(save_path)

# import json
# with open('pokemon_data.json', 'r', encoding='utf-8') as r:
#     pokedex = json.load(r)


class PokemonClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_name = "google/vit-base-patch16-224"  # name of the pretrained model
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            model_name
        )  # resizes and normalizes
        self.batch_size = 16  # use whatever you can fit in memory
        self.lr = 5e-5
        num_pokemon = 150  # total number of pokemon/labels
        self.model.classifier = torch.nn.Linear(
            768, num_pokemon
        )  # Output layer should match the total number of pokemon for classification
        self.model.num_labels = num_pokemon
        # self.update_config()

    def prepare_data(self):
        # train_data = tuple(self.get_datasets('../data_collection/training_data_augmented'))
        pokemon_train_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.ToTensor(),
                # transforms.RandomCrop(32, padding=4),
                # transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                # ),
            ]
        )
        dataset = ImageFolder(
            "C:/study/ee576/project/data/PokemonData", pokemon_train_transforms
        )
        random_seed = 80
        manual_seed(random_seed)

        val_size = floor((len(dataset)) * 0.10)
        train_size = len(dataset) - val_size

        self.train_ds, self.test_ds = random_split(dataset, [train_size, val_size])
        # self.train_ds = random_split(train_data, [len(train_data), 0])[0]
        # # test_data = tuple(self.get_datasets('../data_collection/testing_data'))
        # self.test_ds = random_split(test_data, [len(test_data), 0])[0]

    def forward(self, batch, batch_idx):
        img = batch[0]
        batch[0] = {"pixel_values": img}
        return self.model(batch[0]["pixel_values"].squeeze(), labels=batch[1].squeeze())

    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        # logs = {"train_loss": train_loss}
        # batch_dict = {
        #     "loss": train_loss,
        #     "log": logs,
        # }
        # return batch_dict
        self.log("train_loss: ", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch, batch_idx)
        predicted_labels = output[1].argmax(-1)
        real_labels = batch[1].squeeze()
        accuracy = torch.sum(predicted_labels == real_labels) / self.batch_size

        # tb_logs = {"acc": accuracy}
        # batch_dict = {
        #     "log": tb_logs,
        # }
        # return batch_dict
        self.log("val_accuracy: ", accuracy)
        loss = output[0]
        self.log("val_loss: ", loss)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        # return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, num_workers=2, pin_memory=True)
        # return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=0)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)


class SaveCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        if pl_module.current_epoch >= 0:
            current_epoch = str(pl_module.current_epoch)
            fn = f"epoch_{current_epoch}"
            new_path = f"{save_path}/{fn}/"
            if fn not in os.listdir(save_path):
                os.mkdir(new_path)
            pl_module.model.save_pretrained(new_path)


# Temp hack to surpress logging
import logging

# logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(0)

PokeModel = PokemonClassifier()
logger = TensorBoardLogger("logs", name="pokemon")
trainer = pl.Trainer(
    accumulate_grad_batches=4,
    # precision = 'bf16', # only use if you have an Ampere GPU or newer
    default_root_dir="logs",
    # gpus=1,
    accelerator="gpu",
    devices="auto",
    max_epochs=5,
    callbacks=[SaveCallback()],
    # val_check_interval=1.0,
    check_val_every_n_epoch=1,
    logger=logger,
)

if __name__ == "__main__":
    trainer.fit(PokeModel)

    ### Inference ###
    # model_name = "google/vit-base-patch16-224"  # name of the pretrained model
    # model = ViTForImageClassification.from_pretrained(model_name)
    # model.eval()

    ## Pick an input image

    # x =

    # ## Run inference through model?
    # with torch.no_grad():
    #     y_hat = PokeModel(x)