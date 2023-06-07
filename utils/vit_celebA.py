import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torchvision.transforms import ToTensor

from dataloader import getDataLoader

np.random.seed(0)
torch.manual_seed(0)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

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
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
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


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
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
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )
        
        # Residual?
        # self.to_latent = nn.Identity()

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
        
        # Residual
        # out = self.to_latent(out)
        
        return self.mlp(out)


def main():
    # Loading data
    # train_loader, val_loader, test_loader = getDataLoader("celebA", batch_size=32)
    train_loader, test_loader = getDataLoader("pokemon", batch_size=8)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )

    #(3, 224, 224) 16x16
    model = ViT((3, 224, 224), n_patch=16, n_blocks=4, hidden_d=256, n_heads=8, out_d=150).to(device)

    N_EPOCHS = 25
    LR = 0.0002

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            # y = torch.from_numpy(np.ones(32, dtype=int))
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
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
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
