import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)

## assumptions on inputs
# H == W

def image_to_patch(imgs, n_patch):
    n, c, h, w = imgs.shape # num_imgs, channel, img_w, img_h
    assert h == w # works only for square images
    total_patches = n_patch * n_patch
    patch_size = h // n_patch
    n_pixels = (patch_size * c) ** 2 #HxC/P x WxC/P = (H/P)xC x (W/P)xC ..h==w .. ((H/P)xC)^2
    
    patches = torch.zeros(n, total_patches, n_pixels) #N, 49, 16
    
    for idx, img in enumerate(imgs):
        for i in range(n_patch):
            for j in range(n_patch):
                patch = img[:, i * n_pixels: (i + 1) * n_pixels, j * n_pixels: (j + 1) * n_pixels] #C, H, W
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
        self.hid_d = hid_d
        self.n_heads = n_heads
        assert hid_d % n_heads == 0, f"Can't divide dimension {hid_d} into {n_heads} heads"
        
        dim_per_head = int(hid_d / n_heads)
        self.dim_per_head = dim_per_head

        # Get q, k, v linear mappings
        self.q_map = nn.ModuleList([nn.Linear(dim_per_head, dim_per_head) for _ in range(self.n_heads)])
        self.k_map = nn.ModuleList([nn.Linear(dim_per_head, dim_per_head) for _ in range(self.n_heads)])
        self.v_map = nn.ModuleList([nn.Linear(dim_per_head, dim_per_head) for _ in range(self.n_heads)])
        # Softmax 
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, sequences):

class ViT(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), n_patch=7, hidden_d=8):
        super(ViT, self).__init__()
        
        # Attributes
        self.img_shape = img_shape # (C, H, W)
        self.n_patch = n_patch
        self.hidden_d = hidden_d
        
        assert img_shape[1] % n_patch == 0, "Input shape not entirely divisible by number of patches"
        assert img_shape[2] % n_patch == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (img_shape[1] / n_patch, img_shape[2] / n_patch)
        
        # Linear mapper: map each 16-dimensional patch to an 8-dimensional patch (N, 49, 16) -> (N, 49, 8)
        self.input_d = int(img_shape[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # Add learnable classifiation token to each sequence (N, 49, 8) -> (N, 50, 8)
        self.cls_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # Get positional embeddings for each patch with 50 tokens
        self.pos_embed = nn.Parameter(torch.tensor(embed_position(self.n_patch ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False

    def forward(self, img):
        patches = image_to_patch(img, self.n_patch)
        img_tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        cat_tokens = []
        for i in range(len(tokens)):
            cat_tokens.append(torch.vstack((self.class_token, img_tokens[i])))
        tokens = torch.stack(cat_tokens)
        return tokens

def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = ViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    N_EPOCHS = 5
    LR = 0.005

if __name__ == '__main__':
    main()

##############################
# class ViT(nn.Module):
#     def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', num_ch=3, dimm_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         img_h, img_w = (img_size, img_size) 
#         patch_h, patch_w = (patch_size, patch_size)
        
#         assert img_h % patch_h == 0 and img_w % patch_w == 0, 'Image dimensions must be divisible by the patch size.'
#         n_patches = (img_h // patch_h) * (img_w // patch_w)
        
#         patch_dim = num_ch * patch_h * patch_w
        
#     # def forward(self, img):