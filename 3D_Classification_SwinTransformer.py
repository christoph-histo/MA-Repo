import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import video
from torch.utils.data import DataLoader
import Slice_datacreation

data_path = "/home/histo/Dokumente/christoph/Masterarbeit/Datensätze"

device = torch.device("cuda:0")

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = Slice_datacreation.VolumeToSliceDataset(data_path,transform=data_transform)

train_set, val_set = torch.utils.data.random_split(dataset, [0.9,0.1])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
dataloaders = {'train': train_loader, 'val': val_loader}

model = video.swin3d_b(video.Swin3D_B_Weights.KINETICS400_V1)

num_ftrs = model.head.in_features
model.head = nn.Linear(num_ftrs, 3)  

model = model.to(device)

def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    probs = torch.nn.Softmax(outputs)
                    print(probs)
                    _, preds = torch.max(probs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it achieves the best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set up the optimizer (Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model = train_model(model, criterion, optimizer, num_epochs=25)

torch.save(model.state_dict(), 'resnet_organ_classification_no_aug.pth')