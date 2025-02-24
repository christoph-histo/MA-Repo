import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import video
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/christoph/Dokumente/christoph-MA/MA-Repo')
sys.path.append('/home/christoph/Dokumente/christoph-MA/research-contributions/SwinUNETR')
import Dataloader_patches
import SwinUNETR
from train import train_model
from eval import evaluate_model
from collections import OrderedDict


data_path = "/storage/Datensätze"


model_path = "/home/christoph/Dokumente/christoph-MA//research-contributions/model_swinvit.pt"


state_dict = torch.load(model_path)


# Extract the nested state dictionary if it exists
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']


device = torch.device("cuda:0")


model = SwinUNETR.swin_unetr_base(input_size=(128,128,32),trainable_layers=['all'],in_channels=1,spatial_dims=3)    


model.load_state_dict(state_dict=state_dict, strict=False)


model.swinViT.layers4[0].downsample.reduction = nn.Linear(3072,3)
model.swinViT.layers4[0].downsample.norm= nn.Identity(3,3)


batch_size = 16


def train(model):

    model = nn.DataParallel(model)

    model = model.to(device)    

    dataset = Dataloader_patches.VolumeToPatchesDataset(data_path, transform=None,num_channels=1,test=False,SwinUnetr = True)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    dataloaders = {'train': train_loader, 'val': val_loader}

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25, device="cuda")

    torch.save(model.state_dict(), 'swinUNETR_3D_BTCV_organ_classification_patches_no_aug.pth')

def eval():

    model_path = 'swinUNETR_3D_BTCV_organ_classification_patches_no_aug.pth'
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

    test_dataset = Dataloader_patches.VolumeToPatchesDataset(data_path, transform=None,test=True,num_channels=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    

    metrics = evaluate_model(model, test_loader=test_loader, device=device) 

    organ_labels = {0: "lung", 1: "skin", 2: "intestine"}
    for organ, stats in metrics.items():
        print(f"Organ: {organ_labels[organ]}")
        print(f"  Average Loss: {stats['average_loss']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")

train(model)