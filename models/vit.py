import numpy as np
from tqdm import tqdm, trange
import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from dataloader import getDataLoader

np.random.seed(0)
torch.manual_seed(0)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

## assumptions on inputs
# H == W

def image_to_patch(imgs, n_patch):
    n, c, h, w = imgs.shape  # num_imgs, channel, img_w, img_h
    assert h == w  # works only for square images
    total_patches = n_patch * n_patch
    patch_size = h // n_patch
    n_pixels = c * (patch_size) ** 2
    # HxC/P x WxC/P = (H/P)xC x (W/P)xC ..h==w .. ((H/P)xC)^2

    patches = torch.zeros(n, total_patches, n_pixels)  # N, 49, 16

    for idx, img in enumerate(imgs):
        for i in range(n_patch):
            for j in range(n_patch):
                patch = img[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]  # C, H, W
                # print(patch.shape)
                patches[idx, i * n_patch + j] = patch.flatten()

    return patches


def embed_position(num_tokens, hid_d):
    pos_embed = torch.ones(num_tokens, hid_d)
    for i in range(num_tokens):
        for j in range(hid_d):
            if j % 2 == 0:
                pos_embed[i][j] = np.sin(i / (10000 ** (j / hid_d)))
            else:
                pos_embed[i][j] = np.cos(i / (10000 ** ((j - 1) / hid_d)))
    return pos_embed


class MSA(nn.Module):
    def __init__(self, hid_d, n_heads=2):
        super(MSA, self).__init__()
        self.hid_d = hid_d
        self.n_heads = n_heads
        assert (
            hid_d % n_heads == 0
        ), f"Can't divide dimension {hid_d} into {n_heads} heads"

        dim_per_head = int(hid_d / n_heads)
        self.dim_per_head = dim_per_head

        # Get q, k, v linear mappings
        self.q_map = nn.ModuleList(
            [nn.Linear(dim_per_head, dim_per_head) for _ in range(self.n_heads)]
        )
        self.k_map = nn.ModuleList(
            [nn.Linear(dim_per_head, dim_per_head) for _ in range(self.n_heads)]
        )
        self.v_map = nn.ModuleList(
            [nn.Linear(dim_per_head, dim_per_head) for _ in range(self.n_heads)]
        )
        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_map[head]
                k_mapping = self.k_map[head]
                v_mapping = self.v_map[head]
                seq = sequence[
                    :, head * self.dim_per_head : (head + 1) * self.dim_per_head
                ]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.dim_per_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class TransformerBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.msa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.msa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class ViT(nn.Module):
    def __init__(
        self,
        img_shape=(1, 28, 28),
        n_patch=7,
        hidden_d=8,
        n_heads=2,
        n_blocks=2,
        out_d=10,
    ):
        super(ViT, self).__init__()

        # Attributes
        self.img_shape = img_shape  # (C, H, W)
        self.n_patch = n_patch
        self.hidden_d = hidden_d

        assert (
            img_shape[1] % n_patch == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
            img_shape[2] % n_patch == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (img_shape[1] / n_patch, img_shape[2] / n_patch)

        # Linear mapper: map each 16-dimensional patch to an 8-dimensional patch (N, 49, 16) -> (N, 49, 8)
        self.input_d = int(img_shape[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Add learnable classifiation token to each sequence (N, 49, 8) -> (N, 50, 8)
        self.cls_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Get positional embeddings for each patch with 50 tokens
        self.pos_embed = nn.Parameter(
            torch.tensor(embed_position(self.n_patch**2 + 1, self.hidden_d))
        )
        self.pos_embed.requires_grad = False

        # Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # Classification MLPk
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

    def forward(self, img):
        n, c, h, w = img.shape
        patches = image_to_patch(img, self.n_patch).to(self.pos_embed.device)
        img_tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        cat_tokens = []
        for i in range(len(img_tokens)):
            cat_tokens.append(torch.vstack((self.cls_token, img_tokens[i])))
        tokens = torch.stack(cat_tokens)

        # Adding positional embedding
        out = tokens + self.pos_embed.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)

# Helpers to save Checkpoint
def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'best_result': best_result,
            'best_epoch': best_epoch}

def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


## MAIN ##
def main():
    # Loading data
    train_loader, val_loader, test_loader = getDataLoader("cifar", batch_size=32)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )

    model = ViT( (3, 32, 32), n_patch=8, n_blocks=4, hidden_d=128, n_heads=4, out_d=10).to(device)

    ## Set Training Parameters here 
    N_EPOCHS = 80
    LR = 0.0002
    
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    
    ### Training loop ###
    ckpt_dir = os.path.join('/home/xtreme/runs/ee576-cv-vit/runs', datetime.datetime.now().strftime('%Y%m%d'))
    os.makedirs(ckpt_dir, exist_ok=True)

    best_result = 0
    best_epoch = 0
    train_losses = []
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.detach().cpu().item() / len(train_loader)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_losses.append(train_loss)  
        print(f"\nEpoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
        
        
        ### Validation Loss ###
        correct, total = 0, 0
        val_accs = []
        val_losses = []
        if (epoch % 10) == 0:
            val_loss = 0.0
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                
                loss = criterion(y_hat, y)
                val_loss += loss.detach().cpu().item() / len(val_loader)
                
                
                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)
                val_acc = correct / total
                
                
                if val_acc > best_result:
                    best_result = val_acc
                    best_epoch = epoch
                    # Save best checkpoint
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d' % epoch)
                    save_checkpoint(get_checkpoint_state(model, optimizer, epoch, best_result, best_epoch), ckpt_name)
            
            val_losses.append(val_loss)
            val_accs.append(val_acc)      
            print(f'\t  val_loss: {val_loss:.2f} val_acc: {val_acc:.2f}\n')

    # Save all training outputs
    with open(os.path.join(ckpt_dir, 'train_loss.txt'), 'w') as f:
        for loss in train_losses:
            f.write(str(round(loss, 2)) + '\n')
        f.close()
    with open(os.path.join(ckpt_dir, 'val_loss.txt'), 'w') as f:
        for loss in val_losses:
            f.write(str(round(loss, 2)) + '\n')
        f.close()
    with open(os.path.join(ckpt_dir, 'val_acc.txt'), 'w') as f:
        for acc in val_accs:
            f.write(str(round(acc, 2)) + '\n')
        f.close()

    # Test loop
    print('\n---------------')
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
            test_acc = correct / total
            
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {test_acc * 100:.2f}%")
    


if __name__ == "__main__":
    main()