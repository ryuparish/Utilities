import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing
import copy
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_path = os.path.dirname(os.path.realpath(__file__))

# Getting the imagenames and labels from the csv
df_train = pd.read_csv(input_path + '/english_short.csv')
split = np.random.rand(len(df_train)) < .8
df_val = df_train[~split]
df_train = df_train[split]
encoder = preprocessing.LabelEncoder()
train_labels = df_train['label'].tolist()
encoder.fit(train_labels)
train_labels = encoder.transform(train_labels)
test_labels = encoder.transform(df_val['label'].tolist())
val_images = np.array([np.array(Image.open(input_path + '/' + n)) for n in df_val['image'].tolist()])
train_images = np.array([np.array(Image.open(input_path + '/' + n)) for n in df_train['image'].tolist()])

# Creating the English dataset
class EnglishDataset(Dataset):
    
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, n):
        # While in the dataloader, this will get (batchnumber) images and (batchnumber) labels
        image = self.data[n]
        label = self.labels[n]
        if self.transform:
            image = self.transform(image)
        return (image, label)

# Getting the batch size and the softmax classes
batch_size = 16
classes = range(62)

# Getting the train data set and the batch norm stats
train_mean = np.mean(train_images)/255
train_std = np.mean(val_images)/255

# Defining some transforms
train_transform = transforms.Compose([

    transforms.ToPILImage(),
    transforms.CenterCrop(900, 700),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std])])

val_transform = transforms.Compose([
    
    transforms.ToPILImage(),
    transforms.CenterCrop(900, 700),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std])])

# Dataframes -> Datasets -> Dataloaders
train_dataset = EnglishDataset(train_images, labels=train_labels, transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = EnglishDataset(val_images, labels=test_labels, transform=val_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Making the nn
class EnglishModel(nn.Module):

    def __init__(self):
        super(EnglishModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(.25),)
        self.fc = nn.Sequential(
            nn.Linear(8774912, 62))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# Compiling the model and setting the loss function, optimizer.
model = EnglishModel()
model.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Setting some training variables
epochs = 50
train_losses, val_losses = [], []
train_accu, val_accu = [], []
start_time = time.time()
early_stop_counter = 10
counter = 0
best_val_loss = float('inf')
RETRAIN = True

if RETRAIN is True:
    # Compiling the model and setting the loss function, optimizer.
    model = EnglishModel()
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Creating the training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0
        accuracy = 0
    
        model.train()
    
        print("Epoch: {}/{}.. ".format(epoch+1, epochs))
    
        # For each epoch
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            log_yhat = model(images)
            yhat = torch.exp(log_yhat)
            _, top_class = yhat.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            ### Test this without the cumulative addition, just setting
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            loss = criterion(log_yhat, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Putting the training and accuracy into the list of training accuracies/losses
        train_losses.append(running_loss/len(train_loader))
        train_accu.append(accuracy/len(train_loader))
        val_loss = 0
        accuracy = 0
    
        # Now in evaluation/validation mode
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                ### Try removing the to device
                images = images.to(device)
                labels = labels.to(device)
                log_yhat = model(images)
                val_loss += criterion(log_yhat, labels)
                yhat = torch.exp(log_yhat)
                _, top_class = yhat.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
    
        val_losses.append(val_losses/len(val_loader))
        val_accu.append(accuracy/len(val_loader))
    
        print("Time: {:.2f)s..".format(time.time()-epoch_start_time),
                "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                "Training Accu: {:.3f}.. ".format(train_accu[-1]),
                "Val Loss: {:.3f}.. ".format(val_losses[-1]),
                "Val Accu: {:.3f}..".format(val_accu[-1]))
    
        # Progress check
        if val_losses[-1] < best_val_losses:
            best_val_losses = val_losses[-1]
            counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            print('Validation loss has not imporved since: {:.3f}.. '.format(best_val_loss), 'Count: ', str(counter))
            if counter >= early_stop_counter:
                print('Early Stopping Now!!')
                model.load_state_dict(best_model_wts)
                break
    
    # Saving the trained model
    torch.save(model, '~/kaggle/english_dl/english_model.pt')

else:
    # Compiling the model and setting the loss function, optimizer.
    model = EnglishModel()
    model.load_state_dict(torch.load('~/kaggle/english_dl/english_model.pt'))
    model.to(device)

# plot training history
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
ax = plt.gca()
ax.set_xlim([0, epoch+2])
plt.ylabel('Loss')
plt.plot(range(1, epoch + 2), train_losses[:epoch+1], 'r', label='Training Loss')
plt.plot(range(1, epoch + 2), val_losses[:epoch+1], 'b', label='Validation Loss')
ax.grid(linestyle='solid')
plt.legend()
plt.subplot(2,1,2)
ax = plt.gca()
ax.set_xlim([0, epoch +2])
plt.ylabel('Accuracy')
plt.plot(range(1, epoch + 2), train_accu[:epoch+1], 'r', label='Training Accuracy')
plt.plot(range(1, epoch + 2), val_accu[:epoch+1], 'b', label='Validation Accuracy')
ax.grid(linestyle='solid')
plt.legend()
plt.show()

