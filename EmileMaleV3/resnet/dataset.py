import os
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import pandas as pd
import torchvision.transforms as transforms
import numpy as np

# label 0 for MARY
# label 1 for other class

class ArtDLDataset(Dataset):
  def __init__(self, data_dir = None, transform = None, labels_path = None, set_type = 'train'):

    # Setting the inital_dir to take images from
    self.data_dir = data_dir

    # Setting up the transforms
    self.transform = transform

    # Label path to reads labels_csv from
    self.labels_path = labels_path
    labels_df = pd.read_csv(self.labels_path)

    # Filtering df based on set type
    self.labels_df = labels_df[labels_df['set'] == set_type]
    self.img_names = list(self.labels_df['item'])

  def __getitem__(self, idx):

    # Getting the filename based on idx
    filename = self.img_names[idx]

    # Reading using PIL
    image = Image.open(self.data_dir + "/" + filename + ".jpg")

    # Applying transforms if any
    if(self.transform!=None):
      image = self.transform(image)
    
    # Getting the label 
    # image_label = self.labels_df[self.labels_df['item'] == filename].values.squeeze()[2:20].argmax()
    
    # if image_label > 0:
    #   image_label = 1
    row = self.labels_df[self.labels_df['item'] == filename].values.squeeze()[2:20]
    image_label = 0
    if(row[0]!=1):
      image_label=1

    return (image, image_label, filename)

  def __len__(self):
    return len(self.img_names)


# Util class to apply padding to all the images
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return FT.pad(image, padding, 0, 'constant')

transform=transforms.Compose([
    SquarePad(),

    transforms.Resize((224,224)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(p=0.5)
])


val_transform = transforms.Compose([

	  SquarePad(),
		transforms.Resize(224),
	  transforms.CenterCrop(224),
		transforms.ToTensor()
		
])