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

# Training Data transformation
pokemon_train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Define Image Classifier on Pokemon Dataset
class PokemonClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Pretrain Model
        model_name = "google/vit-base-patch16-224"
        self.model = ViTForImageClassification.from_pretrained(model_name)
        # self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
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

    def forward(self, batch):
        # Expand image shape (3, 244, 244) -> (1, 3, 244, 244)
        x = batch[0][None, :]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log("train_loss: ", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_label = self(batch, batch_idx)
        pred_label = pred_label.argmax(-1)
        gt_label = batch[1].squeeze()
        accuracy = torch.sum(pred_label == gt_label) / self.batch_size
        # Log to tensorboard
        self.log("val_accuracy: ", accuracy)
        self.log("val_loss: ", loss)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, self.batch_size, shuffle=True, num_workers=2, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, num_workers=2, pin_memory=True)


### Inferencing one image ###
def predict_pokemon(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PokemonModel = PokemonClassifier()

    # Load checkpoint
    ckpt_name = "epoch=4-step=480.ckpt"
    model = PokemonModel.load_from_checkpoint(
        os.path.join(PokemonModel.ckpt_dir, ckpt_name)
    )

    # Load Inference Image
    im = Image.open(img_path).convert("RGB")
    data = pokemon_train_transforms(im)
    data = data[None, :].to(device)  # expand dimension

    # Get prediction
    output = model(data)
    predicted_id = output.logits.argmax(-1).item()
    pred_pokemon_name = os.listdir(PokemonModel.data_root)[predicted_id]
    print(f"Predicted Pokemon: {pred_pokemon_name}")

    # Map predicted_id to image
    pred_pokemon_imgs = os.listdir(
        os.path.join(PokemonModel.data_root, pred_pokemon_name)
    )
    pred_img_path = os.path.join(
        PokemonModel.data_root, pred_pokemon_name, pred_pokemon_imgs[0]
    )
    pokemon_img = Image.open(pred_img_path).convert("RGB")
    return pokemon_img, pred_pokemon_name


### Inference ###
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PokemonModel = PokemonClassifier()
    # Set checkpoint path and name
    ckpt_name = "epoch=4-step=480.ckpt"
    model = PokemonModel.load_from_checkpoint(
        os.path.join(
            "C:/study/ee576/project/ee576-cv-vit/models/logs/pokemon/version_4/checkpoints",
            ckpt_name,
        )
    )
    # Pick an input image
    data_root = "C:/study/ee576/project/data/PokemonData"
    pokemon = "Pikachu"
    img_list = os.listdir(os.path.join(data_root, pokemon))
    print(f"Inference Image: {pokemon}")

    # Load pokemon's first image to inference
    img_path = os.path.join(data_root, pokemon, img_list[0])
    img = Image.open(img_path).convert("RGB")
    data = pokemon_train_transforms(img)
    data = data[None, :].to(device)  # expand dimension

    # Run inference through model
    output = model(data)
    predicted_id = output.logits.argmax(-1).item()
    print(f"Predicted ID: {predicted_id}")

    # Map to image
    pokemon_list = os.listdir(data_root)
    pred_pokemon_dir = pokemon_list[predicted_id]
    pred_pokemon = os.listdir(os.path.join(data_root, pred_pokemon_dir))
    pred_img = os.path.join(data_root, pokemon, pred_pokemon[0])
    imgp = Image.open(pred_img).convert("RGB")
    imgp.show()

    ## Debug callable function
    # predict_pokemon(img_path)
