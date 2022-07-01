import os
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import pandas as pd
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
    image_label = self.labels_df[self.labels_df['item'] == filename].values.squeeze()[2:20].argmax()
    
    return (image, image_label)

  def __len__(self):
    return len(self.img_names)