import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from Slice_datacreation import VolumeToSliceDataset

# Define the evaluation function
def evaluate_model(model, test_datasets_path, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define data transforms (same as training)
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

    test_dataset = VolumeToSliceDataset(test_datasets_path, transform=data_transform, test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
    
    # Switch model to evaluation mode
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

# Load the trained model
model_path = 'resnet_organ_classification_no_aug.pth'
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # Assuming 3 classes: lung, skin, intestine
model.load_state_dict(torch.load(model_path))

# Evaluate the model on the test datasets
test_data_path = "/home/histo/Dokumente/christoph/Masterarbeit/Datensätze"
metrics = evaluate_model(model, test_data_path)

# Print the results
organ_labels = {0: "lung", 1: "skin", 2: "intestine"}
for organ, stats in metrics.items():
    print(f"Organ: {organ_labels[organ]}")
    print(f"  Average Loss: {stats['average_loss']:.4f}")
    print(f"  Accuracy: {stats['accuracy']:.4f}")

