import sys
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
sys.path.append('/home/christoph/Dokumente/christoph-MA/MA-Repo')
import Dataloader_whole_slices
import Dataloader_slice_parts
from train import train_model   
from eval import evaluate_model 
from collections import OrderedDict

data_path = "/storage/Datensätze"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

data_transform = transforms.Compose([
    transforms.Resize((224*4)),
    transforms.ToTensor(),
])

model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)

num_ftrs = model.head.in_features
model.head = nn.Linear(num_ftrs, 3)

batch_size = 16

def train():

    global model

    model = nn.DataParallel(model)

    model = model.to(device)    

    dataset = Dataloader_slice_parts.VolumeToSlicepartsDataset(data_path, transform=None,test=True)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    dataloaders = {'train': train_loader, 'val': val_loader}

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25, device="cuda")

    torch.save(model.state_dict(), '/home/christoph/Dokumente/christoph-MA/Models/swin_transformer_2D_organ_classification_slice_parts_no_aug.pth')

def eval():

    global model
    
    model_path = '/home/christoph/Dokumente/christoph-MA/Models/swin_transformer_2D_organ_classification_slice_parts_no_aug.pth'
    state_dict = torch.load(model_path)

    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v

    # Load the modified state dictionary into the model
    model.load_state_dict(new_state_dict)

    test_dataset = Dataloader_whole_slices.VolumeToSliceDataset(data_path, transform=data_transform, test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    

    metrics = evaluate_model(model, test_loader = test_loader, device = device) 

    organ_labels = {0: "lung", 1: "skin", 2: "intestine"}
    for organ, stats in metrics.items():
        print(f"Organ: {organ_labels[organ]}")
        print(f"  Average Loss: {stats['average_loss']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")

eval()