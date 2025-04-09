import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import video
from torch.utils.data import DataLoader
import sys
import os
from collections import OrderedDict

sys.path.append('/home/christoph/Dokumente/christoph-MA/MA-Repo')
import Dataloader_patches
from train import train_model
from eval import evaluate_model


def train(data_path, model, save_path, device, augmentation):

    batch_size = 32

    model = nn.DataParallel(model)
    model = model.to(device)

    dataset = Dataloader_patches.VolumeToPatchesDataset(root_dir=data_path, test=False, augmentation=augmentation)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    dataloaders = {'train': train_loader, 'val': val_loader}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25, device=device)

    torch.save(model.state_dict(), save_path)


def eval(data_path, model, model_path, device):
    batch_size = 32

    state_dict = torch.load(model_path)

    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load the modified state dictionary into the model
    model.load_state_dict(new_state_dict)

    dataset = Dataloader_patches.VolumeToPatchesDataset(root_dir=data_path, test=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    metrics = evaluate_model(model, test_loader=test_loader, device=device)

    organ_labels = {0: "lung", 1: "skin", 2: "intestine"}
    for organ, stats in metrics.items():
        print(f"Organ: {organ_labels[organ]}")
        print(f"  Average Loss: {stats['average_loss']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")

    # Save accuracies in a .csv file
    with open(f'/home/christoph/Dokumente/christoph-MA/Models/metrics_{os.path.basename(model_path)}.csv', 'w') as f:
        f.write("Organ,Average Loss,Accuracy\n")
        for organ, stats in metrics.items():
            f.write(f"{organ_labels[organ]},{stats['average_loss']:.4f},{stats['accuracy']:.4f}\n")


def setup(mode="train", augmentation="no_aug"):
    data_path = "/storage/Datensätze"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    model = video.r2plus1d_18(weights=video.R2Plus1D_18_Weights.KINETICS400_V1)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)

    save_path = f'/home/christoph/Dokumente/christoph-MA/Models/resnet_3D_organ_classification_patches_{augmentation}.pth'

    if augmentation == "no_aug":
        aug = None
    else:
        aug = augmentation

    if mode == "train":
        train(data_path=data_path, model=model, save_path=save_path, device=device, augmentation=aug)
    elif mode == "eval":
        eval(data_path=data_path, model=model, model_path=save_path, device=device)
    else:
        print("Error: mode not supported")
        return
    
if __name__ == "__main__":
    #setup(mode="train", augmentation="no_aug")
    setup(mode="eval", augmentation="no_aug")