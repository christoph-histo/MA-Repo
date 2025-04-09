import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.models import video
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/christoph/Dokumente/christoph-MA/MA-Repo')
import Aggregator_Module
import Dataloader_patches_aggregator
from train import train_model
from eval import evaluate_model
from collections import OrderedDict

data_path = "/storage/Datensätze"

device = torch.device("cuda")

encoder = video.swin3d_b(video.Swin3D_B_Weights.KINETICS400_V1)

num_ftrs = encoder.head.out_features

encoder.to(device)

dropout = 0.1

decoder_enc = nn.Sequential(
                            nn.Linear(num_ftrs, 128),
                            nn.GELU(),
                            nn.Dropout(dropout)
                            )

model = Aggregator_Module.AttnMeanPoolMIL(gated=True, dropout=dropout, out_dim=3,encoder=decoder_enc,encoder_dim=128)

model.start_attention(freeze_encoder=False)

batch_size = 8

epochs = 100

def train():

    global model, encoder

    model = nn.DataParallel(model)

    model = model.to(device)    

    dataset = Dataloader_patches_aggregator.VolumeToFeaturesDataset(data_path, transform=None,num_channels=3, test=False,encoder=encoder)

    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    dataloaders = {'train': train_loader, 'val': val_loader}

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0005)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=epochs, device="cuda",aggregation=True, scheduler=scheduler)

    torch.save(model.state_dict(), '/home/christoph/Dokumente/christoph-MA/Models/SwinTransformer_Aggregator_3D_organ_classification_patches_no_aug.pth')

def eval():
    
    global model

    model_path = '/home/christoph/Dokumente/christoph-MA/Models/SwinTransformer_Aggregator_3D_organ_classification_patches_no_aug.pth'
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

    test_dataset =  Dataloader_patches_aggregator.VolumeToFeaturesDataset(data_path, transform=None,test=True,encoder=encoder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    

    metrics = evaluate_model(model, test_loader=test_loader, device=device) 

    organ_labels = {0: "lung", 1: "skin", 2: "intestine"}
    for organ, stats in metrics.items():
        print(f"Organ: {organ_labels[organ]}")
        print(f"  Average Loss: {stats['average_loss']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")

train()