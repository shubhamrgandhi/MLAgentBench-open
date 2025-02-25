import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Add data augmentation transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    # Set device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Define the dataloaders 
    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,num_workers=4)

    # Define the optimizer and loss function
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,  momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    def test_model(dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    # Hyperparameter Tuning
    param_grid = {
        'lr': [0.01, 0.05, 0.1, 0.5],
        'momentum': [0.5, 0.7, 0.9],
        'batch_size': [32, 64, 128, 256]
    }

    # Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(train_dataset, train_dataset.targets)
    print("Grid Search Best Parameters:", grid_search.best_params_)

    # Random Search
    param_distributions = {
        'lr': np.logspace(-2, -1, 5),
        'momentum': np.linspace(0.5, 0.9, 5),
        'batch_size': [32, 64, 128, 256]
    }

    random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, n_jobs=-1)
    random_search.fit(train_dataset, train_dataset.targets)
    print("Random Search Best Parameters:", random_search.best_params_)

    # Retrain the model with the best hyperparameters from grid search
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=grid_search.best_params_['lr'], momentum=grid_search.best_params_['momentum'])
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        train_accuracy = test_model(train_dataloader)
        test_accuracy = test_model(test_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

    # print training accuracy
    train_accuracy = test_model(train_dataloader)
    test_accuracy = test_model(test_dataloader)
    print (f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

    # Save the predictions to submission.csv
    import pandas as pd
    submission = pd.DataFrame(columns=list(range(10)), index=range(len(test_dataset)))
    model.eval()
    for idx, data in enumerate(test_dataset):
        inputs = data[0].unsqueeze(0).to(device)
        pred = model(inputs)
        pred = torch.softmax(pred[0], dim=0)
        submission.loc[idx] = pred.tolist()
    submission.to_csv('submission.csv')