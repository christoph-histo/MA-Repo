import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel, Sequential, AdaptiveAvgPool3d, Linear
from collections import OrderedDict
import sys
import os

sys.path.append('/home/christoph/Dokumente/christoph-MA/MA-Repo')
sys.path.append('/home/christoph/Dokumente/christoph-MA/MedicalNet/models')
import resnet
import Dataloader_patches
from train import train_model
from eval import evaluate_model


def train(data_path, model, save_path, device, augmentation):
    batch_size = 32

    model = model.to(device)

    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")
    dataset = Dataloader_patches.VolumeToPatchesDataset(root_dir=data_path, transform=None, num_channels=1, test=False, augmentation=augmentation,SwinUnetr=True)

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
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    """
    # Load the modified state dictionary into the model
    model.load_state_dict(state_dict)

    model.to(device)

    dataset = Dataloader_patches.VolumeToPatchesDataset(root_dir=data_path, transform=None, num_channels=1, test=True,SwinUnetr=True)
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

    model = resnet.resnet18(sample_input_D=32, sample_input_H=128, sample_input_W=128, num_seg_classes=1)

    model = nn.DataParallel(model)

    model_path = "/home/christoph/Dokumente/christoph-MA/MedicalNet/pretrain/resnet_18_23dataset.pth"

    model.load_state_dict(torch.load(model_path),strict=False)

    model.module.conv_seg = Sequential(
        AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        nn.Flatten(),
        Linear(in_features=512, out_features=3, bias=True)
    )

    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")

    save_path = f'/home/christoph/Dokumente/christoph-MA/Models/resnet_3D_Med3D_organ_classification_patches_{augmentation}.pth'

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