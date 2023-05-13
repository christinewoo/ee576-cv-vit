from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle
from os import listdir
from os.path import isfile, join
def unpickle(directory):
    train = dict()
    val = dict()
    test = dict()
    for file in listdir(directory):
        if isfile(join(directory, file)):
            with open(directory + '/' + file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')


class celebA(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path ="/home/xtreme/runs/data/celeba/img/" + str(self.file_list[idx])
        img = Image.open(img_path)
        return img

class cifar(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path ="/home/xtreme/runs/data/celeba/img/" + str(self.file_list[idx])
        img = Image.open(img_path)
        return img

def getTrainDataset(dataset_name):
    if dataset_name == "celebA":
        list_path = "/home/xtreme/runs/ee576-cv-vit/splits/train.txt"
        with open(list_path, 'r') as f:
            lines = f.readlines()
            file_list = [line.strip() for line in lines]
            train_data = celebA(file_list=file_list)
            return train_data
    


def getValDataset(dataset_name):
    if dataset_name == "celebA":
        list_path = "/home/xtreme/runs/ee576-cv-vit/splits/val.txt"
        with open(list_path, 'r') as f:
            lines = f.readlines()
            file_list = [line.strip() for line in lines]
            val_data = celebA(file_list=file_list)
            return val_data

def getTestDataset(dataset_name):
    if dataset_name == "celebA":
        list_path = "/home/xtreme/runs/ee576-cv-vit/splits/test.txt"
        with open(list_path, 'r') as f:
            lines = f.readlines()
            file_list = [line.strip() for line in lines]
            test_data = celebA(file_list=file_list)
            return test_data


# def test():
#     dataset = getValDataset()
#     img = dataset.__getitem__(0)
#     img.save("val.jpg")
#     d2 = getTrainDataset()
#     img = d2.__getitem__(0)
#     img.save("train.jpg")

# test()

c = unpickle("/home/xtreme/runs/data/cifar-10-batches-py/data_batch_1")
print(c.keys())