import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import video
from torch.utils.data import DataLoader
from torch.nn import DataParallel, Sequential, AdaptiveAvgPool3d, Linear
import sys
sys.path.append('/home/christoph/Dokumente/christoph-MA/MA-Repo')
sys.path.append('/home/christoph/Dokumente/christoph-MA/MedicalNet/models')
import resnet
import Dataloader_patches
from train import train_model
from eval import evaluate_model
from collections import OrderedDict

model_path = "/home/christoph/Dokumente/christoph-MA/MA-Repo"

device = torch.device("cuda")

data_path = "/storage/Datensätze"

model = resnet.resnet18(sample_input_D=32, sample_input_H=128, sample_input_W=128, num_seg_classes=1)

model_path = "/home/christoph/Dokumente/christoph-MA/MedicalNet/pretrain/resnet_18_23dataset.pth"

model.load_state_dict(torch.load(model_path),strict=False)

model = nn.DataParallel(model)

model.module.conv_seg = Sequential(
    AdaptiveAvgPool3d(output_size=(1, 1, 1)),
    nn.Flatten(),
    Linear(in_features=512, out_features=3, bias=True)
)

batch_size = 32

def train():
    global model

    model = nn.DataParallel(model)

    model = model.to(device)    

    dataset = Dataloader_patches.VolumeToPatchesDataset(data_path, transform=None,num_channels=3)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    dataloaders = {'train': train_loader, 'val': val_loader}

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25, device="cuda")

    torch.save(model.state_dict(), 'resnet_Med3D_organ_classification_patches_no_aug.pth')

def eval():

    model_path = 'resnet_Med3D_organ_classification_patches_no_aug.pth'
    state_dict = torch.load(model_path)

    # Load the modified state dictionary into the model
    model.load_state_dict(state_dict)

    test_dataset = Dataloader_patches.VolumeToPatchesDataset(data_path, transform=None,test=True,num_channels=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    

    metrics = evaluate_model(model, test_loader=test_loader, device=device) 

    organ_labels = {0: "lung", 1: "skin", 2: "intestine"}
    for organ, stats in metrics.items():
        print(f"Organ: {organ_labels[organ]}")
        print(f"  Average Loss: {stats['average_loss']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")

train()