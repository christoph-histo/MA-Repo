import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import Dataloader_whole_slices 
from collections import OrderedDict

# Define the evaluation function
def evaluate_model(model, test_loader, device="cuda"):

    device = device
    print("device: ", device) 
    
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Organ-wise tracking
    organ_correct = {0: 0, 1: 0, 2: 0}  # Correct predictions per organ
    organ_total = {0: 0, 1: 0, 2: 0}    # Total samples per organ
    organ_loss = {0: 0.0, 1: 0.0, 2: 0.0}  # Cumulative loss per organ
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            for label, pred in zip(labels, preds):
                organ = label.item()
                organ_total[organ] += 1
                organ_correct[organ] += (pred.item() == organ)
                organ_loss[organ] += loss.item() 

    # Calculate average loss and accuracy per organ
    organ_metrics = {}
    for organ in organ_correct:
        avg_loss = organ_loss[organ] / organ_total[organ] if organ_total[organ] > 0 else 0
        accuracy = organ_correct[organ] / organ_total[organ] if organ_total[organ] > 0 else 0
        organ_metrics[organ] = {'average_loss': avg_loss, 'accuracy': accuracy}
    
    return organ_metrics