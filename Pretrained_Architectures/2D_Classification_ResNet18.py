import sys
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
import os


def train(data_path,model,transforms,save_path,device,augmentation,dataset):

    batch_size = 64

    model = nn.DataParallel(model)

    model = model.to(device)

    if dataset == "slice_parts":
        dataset = Dataloader_slice_parts.VolumeToSlicepartsDataset(data_path=data_path, transform=transforms, test=False, augmentation=augmentation)
    elif dataset == "whole_slices":
        dataset = Dataloader_whole_slices.VolumeToSliceDataset(data_path=data_path, transform=transforms, test=False, augmentation=augmentation)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    dataloaders = {'train': train_loader, 'val': val_loader}

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10, device="cuda")

    torch.save(model.state_dict(), save_path)


def eval(data_path,model,transforms,model_path,device,dataset):

    batch_size = 64

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
    if dataset == "slice_parts":
        test_dataset = Dataloader_slice_parts.VolumeToSlicepartsDataset(data_path, transform=transforms,test=True)
    elif dataset == "whole_slices":
        test_dataset = Dataloader_whole_slices.VolumeToSliceDataset(data_path, transform=transforms,test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    

    metrics = evaluate_model(model, test_loader=test_loader, device=device) 

    organ_labels = {0: "lung", 1: "skin", 2: "intestine"}
    for organ, stats in metrics.items():
        print(f"Organ: {organ_labels[organ]}")
        print(f"  Average Loss: {stats['average_loss']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")

    #save accuracies in a .csv file in /home/christoph/Dokumente/christoph-MA/Models
    with open(f'/home/christoph/Dokumente/christoph-MA/Models/metrics_{model_path}.csv', 'w') as f:
        f.write("Organ,Average Loss,Accuracy\n")
        for organ, stats in metrics.items():
            f.write(f"{organ_labels[organ]},{stats['average_loss']:.4f},{stats['accuracy']:.4f}\n")


def setup(mode = "train",pretrained = True, data_transform = None, augmentation = "no_aug",dataset = "slice_parts"):

    data_path = "/storage/Datensätze"

    if data_transform != None:
        data_transform = transforms.Compose([
            transforms.Resize((224*6)),
            transforms.ToTensor(),
        ])

    else:

        transforms = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device: ", device)

    if pretrained:
        suffix = ""
        weights = models.ResNet18_Weights.DEFAULT
    else:
        suffix = "_not_pretrained"
        weights = None

    model = models.resnet18(weights=weights)

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 3)

    save_path = f'/home/christoph/Dokumente/christoph-MA/Models/resnet_2D_organ_classification_{dataset}_{augmentation}{suffix}.pth'

    if mode == "train":
        train(data_path=data_path, model=model, transforms=transforms, save_path=save_path,device=device,augmentation=augmentation,dataset=dataset)
    elif mode == "eval":
        eval(data_path=data_path, model=model, transforms=transforms, model_path=save_path,device=device,dataset=dataset)
    else:
        print("Error: mode not supported")
        return


