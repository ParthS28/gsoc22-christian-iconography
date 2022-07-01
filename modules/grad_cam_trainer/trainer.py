import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch 
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import ArtDLClassifier
from transforms import transform, val_transform
from dataset import ArtDLDataset

# Dataset
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

# Train

# Set up sampling weights
# y_train_indices = range(len(train_dataset))

# y_train = [train_dataset[i][1] for i in y_train_indices]

# class_sample_count = np.array(
#     [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

# weight = 1. / class_sample_count
# samples_weight = np.array([weight[t] for t in y_train])
# samples_weight = torch.from_numpy(samples_weight)

# sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

device = 'cuda' if torch.cuda.is_available() else "cpu"
print('--------------, ', device)
clf = ArtDLClassifier(num_classes = 19).to(device)
optimizer = optim.SGD(clf.trainable_params(), lr = 0.01, momentum = 0.9)
criterion = nn.CrossEntropyLoss()

clf.load_state_dict(torch.load("artDLresnet50_224x224_5.pt", map_location = device))

def train(epochs, model, train_loader, val_loader, optimizer, device, criterion):
  for epoch in range(epochs):
    # Setting the train mode
    model.train()
    train_loss = 0
    val_loss = 0
    for idx, (image, label) in enumerate(tqdm(train_loader)):
      image = image.to(device)
      label = label.to(device)

      # Zeroing the gradients before re-computing them
      optimizer.zero_grad()
      outputs = model(image).squeeze()
      
      # Calculating the loss
      loss = criterion(outputs, label)
      train_loss += loss.item()

      # Calculating the gradients == diff(loss w.r.t weights)
      loss.backward()

      # Updating the weights
      optimizer.step()
    
    model.eval()
    val_score = 0
    for idx, (image, label) in enumerate(val_loader):
      image = image.to(device)
      label = label.to(device)
      outputs = model(image).squeeze()

      # Getting the predictions
      pred = outputs.argmax(dim = 1, keepdim = True)

      # Updating scores and losses
      val_score += pred.eq(label.view_as(pred)).sum().item()
      loss = criterion(outputs, label)
      val_loss += loss.item()
    
    print("=================================================")
    print("Epoch: {}".format(epoch+1))
    print("Validation Loss: {}".format(val_loss/len(val_loader)))
    print("Training Loss: {}".format(train_loss/len(train_loader)))
    print(val_score)
    print("Validation Accuracy: {}".format((val_score)/len(val_loader)*10))
  torch.save(model.state_dict(), f'artDLresnet50_224x224__{epoch}.pt')

train(50, clf, train_loader, val_loader, optimizer, device, criterion)

# torch.save(clf.state_dict(), 'artDLresnet50_224x224.pt')