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
model.load_state_dict(torch.load("resnet/artDLresnet50_224x224.pt", map_location = device))


dataset = ArtDLDataset(
    data_dir = 'data/images',
    transform = transform,
    labels_path = 'data/info2.csv',
    set_type = 'train'
)

dataloader = DataLoader(dataset = dataset, shuffle=True, batch_size = 1)

df = pd.DataFrame(columns=['item', 'label', 'predicted'])
for idx, (image, label, fname) in enumerate(tqdm(dataloader)):
    image = image.to(device)
    label = label.to(device)
    outputs = model(image).squeeze()
# torch.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    pred = outputs.argmax(dim = -1, keepdim = True)

    # if pred != label:
    df.loc[df.shape[0]] = [fname[0], label.item(), pred.item()]

df.to_csv('out1.csv')
