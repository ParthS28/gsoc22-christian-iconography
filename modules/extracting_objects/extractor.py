import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import cv2 
from PIL import Image
import numpy as np
import pandas as pd

from model import ArtDLClassifier
from utils import roi, get_gradcam
from dataset import ArtDLDataset, transform, val_transform

device = "cpu"

# Load model
model = ArtDLClassifier(num_classes = 19).to(device)
model.load_state_dict(torch.load("artDLresnet50_224x224_8.pt", map_location = device))

# Load data
train_dataset = ArtDLDataset(
    data_dir = 'data/DEVKitArt/JPEGImages',
    transform = transform,
    labels_path = 'data/DEVKitArt/info.csv',
    set_type = 'train'
)

test_dataset = ArtDLDataset(
    data_dir = 'data/DEVKitArt/JPEGImages',
    transform = val_transform,
    labels_path = 'data/DEVKitArt/info.csv',
    set_type = 'test'
)

val_dataset = ArtDLDataset(
    data_dir = 'data/DEVKitArt/JPEGImages',
    transform = val_transform,
    labels_path = 'data/DEVKitArt/info.csv',
    set_type = 'val'
)

train_loader = DataLoader(dataset = train_dataset, shuffle=True, batch_size = 50)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1)
val_loader = DataLoader(dataset = val_dataset, batch_size = 10)

classes = ["MARY",
"ANTONY ABBOT" ,
"ANTONY OF PADUA",
"AUGUSTINE",
"DOMINIC",
"FRANCIS",
"JEROME",
"JOHN THE BAPTIST",
"JOHN",
"JOSEPH",
"PAUL",
"PETER",
"SEBASTIAN",
"STEPHEN",
"BARBARA",
"CATHERINE",
"MARY MAGDALENE",
"John Baptist - Child",
"John Baptist - Dead"
]

df = pd.DataFrame(columns=['object_id', 'item', 'associated_class', 'actual_class'])
def extract(dataloader):
    for idx, (image, label, fname) in enumerate(tqdm(dataloader)):
        
        image = image.to(device)
        label = label.to(device)

        outputs = model(image).squeeze()
        curr = 0
        for i in range(len(outputs)):
            o = outputs[i]
            
            if o > 0:
                associated_class = classes[i]
                heatmap, sm_heatmap = get_gradcam(model, image, o, size=224)

                regions = roi(sm_heatmap, 14)
                
                cur = 0
                padding = 15
                for r in regions:
                    left = max(0, r[0][0]-padding)
                    right = min(223, r[1][0]+padding)
                    top = max(0, r[0][1]-padding)
                    bottom = min(223, r[1][1]+padding)
                    obj = image[0:1, 0:3, left:right, top:bottom]
                    
                    obj = obj[0]
                    obj = obj*255
                    obj = obj.permute(1, 2, 0).numpy().astype(np.uint8)
                    im = Image.fromarray(obj)
                    im.save(f'outputs/out{idx}_{curr}_{cur}.png')
                    df.loc[df.shape[0]] = [f'out{idx}_{curr}_{cur}.png', fname[0], associated_class, classes[label]]
                    cur+=1
            curr+=1
        # break

extract(test_loader)
df.to_csv('outputs/info.csv')
