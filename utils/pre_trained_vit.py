import json
from PIL import Image
import torch
from torchvision import transforms

# Load ViT
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)
model.eval()

# Load image
# NOTE: Assumes an image `img.jpg` exists in the current directory
img = transforms.Compose([
    transforms.Resize((384, 384)), 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])(Image.open('/home/xtreme/runs/data/celeba/075269.jpg')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 384, 384])

labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
with torch.no_grad():
    outputs = model(img)
print(outputs.shape)
print('-----')
for idx in torch.topk(outputs, k=3).indices.tolist():
    prob = torch.softmax(outputs, -1)[idx].item()
    print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))