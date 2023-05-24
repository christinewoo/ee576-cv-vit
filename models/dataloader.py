from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np
import pandas as pd

class celebA(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.label_list = label_list.values

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path ="/home/xtreme/runs/data/celeba/img/" + str(self.file_list[idx])
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label_index = self.file_list[idx].split('.')
        label_index = label_index[0]
        label_index = int(label_index)
        label = self.label_list[label_index-1, 1]
        return img_transformed, label

# Input: the name of dataset (celebA or cifar)
# Output: the Dataset of specific dataset
# Notice that the cifar only provide the train and test dataset, so the validation set should be
# divide after the user get the Dataset.
def getTrainDataset(dataset_name, transform):
    if dataset_name == "celebA":
        list_path = "/home/xtreme/runs/ee576-cv-vit/splits/train.txt"
        with open(list_path, 'r') as f:
            lines = f.readlines()
            file_list = [line.strip() for line in lines]
            label_list = pd.read_csv("/home/xtreme/runs/data/celeba/anno/identity_CelebA.txt", sep=" ", header=None)
            train_data = celebA(file_list=file_list, label_list = label_list, transform=transform)
            return train_data
    else:
        dataset = CIFAR10(root='/home/xtreme/runs/data/cifar10/', download=True, transform=transform)
        return dataset
    

# Notice that the cifar doesn't provide validation set, so the user should split by themselves.
def getValDataset(dataset_name, transform):
    if dataset_name == "celebA":
        list_path = "/home/xtreme/runs/ee576-cv-vit/splits/val.txt"
        with open(list_path, 'r') as f:
            lines = f.readlines()
            file_list = [line.strip() for line in lines]
            label_list = pd.read_csv("/home/xtreme/runs/data/celeba/anno/identity_CelebA.txt", sep=" ", header=None)
            val_data = celebA(file_list=file_list, label_list = label_list, transform=transform)
            return val_data
    else:
        return None

def getTestDataset(dataset_name, transform):
    if dataset_name == "celebA":
        list_path = "/home/xtreme/runs/ee576-cv-vit/splits/test.txt"
        with open(list_path, 'r') as f:
            lines = f.readlines()
            file_list = [line.strip() for line in lines]
            label_list = pd.read_csv("/home/xtreme/runs/data/celeba/anno/identity_CelebA.txt", sep=" ", header=None)
            test_data = celebA(file_list=file_list, label_list = label_list, transform=transform)
            return test_data
    else:
        dataset = CIFAR10(root='/home/xtreme/runs/data/cifar10/', train=False, transform=transform)
        return dataset

def getDataLoader(dataset_name, batch_size=1):
    if dataset_name == "celebA":
        celebA_train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(178), # can try padding later
                transforms.Resize((224, 224)),
                transforms.ToTensor(), # maybe can add norm for convergence
            ]
        )
        celebA_test_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(178),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = getTrainDataset(dataset_name, celebA_train_transforms)
        val_dataset = getValDataset(dataset_name, celebA_test_transforms)
        test_dataset = getTestDataset(dataset_name, celebA_test_transforms)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # mod
        return train_loader, val_loader, test_loader
    else:
        cifar_train_transforms = transforms.Compose(
            [
                # transforms.Resize((224, 224)),
                # transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        cifar_test_transforms = transforms.Compose(
            [
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )
        train_dataset = getTrainDataset(dataset_name, cifar_train_transforms)
        test_dataset = getTestDataset(dataset_name, cifar_test_transforms)
        val_size = 5000
        train_size = len(train_dataset) - val_size
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, test_loader


# def test():
#     train = getTrainDataset("cifar")
#     test = getTestDataset("cifar")
#     val_size = 5000
#     train_size = len(train) - val_size
#     train_ds, val_ds = random_split(train, [train_size, val_size])


# test()

