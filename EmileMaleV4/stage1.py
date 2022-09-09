from resnet.model import ArtDLClassifier
from resnet.dataset import ArtDLDataset, transform, val_transform

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2 
from PIL import Image
import numpy as np


device = 'cuda' if torch.cuda.is_available() else "cpu"

# Load model
model = ArtDLClassifier(num_classes = 2).to(device)
model.load_state_dict(torch.load("resnet/artDLresnet50_224x224_2c_moredata_3.pt", map_location = device))

dataset = ArtDLDataset(
    data_dir = 'data/images',
    transform = val_transform,
    # labels_path = 'data/info_small.csv',
    set_type = 'train'
)

dataloader = DataLoader(dataset = dataset, shuffle=True, batch_size = 1)

model.eval()
df = pd.DataFrame(columns=['item', 'predicted'])
for idx, (image, label, fname) in enumerate(tqdm(dataloader)):
    image = image.to(device)
    label = label.to(device)
    outputs = model(image).squeeze()
    pred = outputs.argmax(dim = -1, keepdim = True)

    df.loc[df.shape[0]] = [fname[0], pred.item()]

df.to_csv('out1.csv')
