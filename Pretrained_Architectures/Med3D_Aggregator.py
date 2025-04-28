import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import video
from collections import OrderedDict
import sys
import os

sys.path.append('/home/christoph/Dokumente/christoph-MA/MA-Repo')
sys.path.append('/home/christoph/Dokumente/christoph-MA/MedicalNet/models')
import resnet
import Aggregator_Module
import Dataloader_patches_aggregator
from train import train_model
from eval import evaluate_model


def train(data_path, model, encoder, save_path, device, augmentation):
    batch_size = 4
    epochs = 50

    model = model.to(device)

    dataset = Dataloader_patches_aggregator.VolumeToFeaturesDataset(data_path, transform=None, num_channels=1, test=False ,SwinUnetr=True, encoder=encoder, augmentation=augmentation)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    dataloaders = {'train': train_loader, 'val': val_loader}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=epochs, device=device, aggregation=True, scheduler=scheduler)

    torch.save(model.state_dict(), save_path)


def eval(data_path, model, encoder, model_path, device):
    batch_size = 4

    state_dict = torch.load(model_path)

    # Load the modified state dictionary into the model
    model.load_state_dict(state_dict)

    test_dataset = Dataloader_patches_aggregator.VolumeToFeaturesDataset(data_path, transform=None, num_channels=1,SwinUnetr=True, test=False, encoder=encoder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    # Initialize the encoder
    encoder = resnet.resnet18(sample_input_D=32, sample_input_H=128, sample_input_W=128, num_seg_classes=1)

    # Modify the model's final layers
    encoder = nn.DataParallel(encoder)

    model_path = "/home/christoph/Dokumente/christoph-MA/MedicalNet/pretrain/resnet_18_23dataset.pth"

    encoder.load_state_dict(torch.load(model_path),strict=False)

    encoder.module.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        nn.Flatten()
    )

    encoder.to(device)

    dropout = 0.1

    # Define the decoder
    decoder_enc = nn.Sequential(
        nn.Linear(512, 128),
        nn.GELU(),
        nn.Dropout(dropout)
    )

    # Initialize the aggregator model
    model = Aggregator_Module.AttnMeanPoolMIL(gated=True, dropout=dropout, out_dim=3, encoder=decoder_enc, encoder_dim=128)
    model.start_attention(freeze_encoder=False)

    save_path = f'/home/christoph/Dokumente/christoph-MA/Models/Med3D_Aggregator_3D_organ_classification_patches_{augmentation}.pth'

    if augmentation == "no_aug":
        aug = None
    else:
        aug = augmentation

    if mode == "train":
        train(data_path=data_path, model=model, encoder=encoder, save_path=save_path, device=device, augmentation=aug)
    elif mode == "eval":
        eval(data_path=data_path, model=model, encoder=encoder, model_path=save_path, device=device)
    else:
        print("Error: mode not supported")
        return


if __name__ == "__main__":
    # Example usage
    setup(mode="train", augmentation="no_aug")
    # setup(mode="eval", augmentation="no_aug")