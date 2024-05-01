import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from customDataset import *
from model import *


EPOCHS = 10
ALPHA = 0.001
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

boston = CityImageSet("boston")
amsterdam = CityImageSet("amsterdam")
paris = CityImageSet("paris")
phoenix = CityImageSet("phoenix")
toronto = CityImageSet("toronto")
zurich = CityImageSet("zurich")

# Combine all city datasets
all_cities = torch.utils.data.ConcatDataset(
    [boston, amsterdam, paris, phoenix, toronto, zurich])

# Split dataset into train and test sets
train_size = int(0.7 * len(all_cities))
test_size = len(all_cities) - train_size
train_dataset, test_dataset = random_split(all_cities, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = MultiCityClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

model.to(device)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# Test Function


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct_predictions = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['image'].to(device), data['city'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = correct_predictions / len(test_loader.dataset)

    return avg_loss, accuracy


# Training Loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1} / {EPOCHS}")
    model.train()
    running_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i, data in enumerate(train_loader):
        print(f"\rIteration {i + 1} / {len(train_loader)}", end="")
        inputs, labels = data['image'].to(device), data['city'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    train_loss.append(running_loss / len(train_loader.dataset))
    train_accuracy.append(correct_predictions / total_predictions)

    # Run full test set
    avg_loss, accuracy = test(model, test_loader, device)
    test_loss.append(avg_loss)
    test_accuracy.append(accuracy)

    print(
        f"\nTrain Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_accuracy[-1]:.4f}")
    print(
        f"Test Loss: {test_loss[-1]:.4f}, Test Accuracy: {test_accuracy[-1]:.4f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"models/cityMultiClassifier_Epoch{epoch}.pth")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("cityMultiClassifier.png")

# First Run
# Device:  cuda
# Epoch 1 / 10
# Iteration 759 / 759
# Train Loss: 9.4975, Train Accuracy: 0.6761
# Test Loss: 0.0612, Test Accuracy: 0.7845
# Epoch 2 / 10
# Iteration 759 / 759
# Train Loss: 0.0221, Train Accuracy: 0.9027
# Test Loss: 0.0570, Test Accuracy: 0.8072
# Epoch 3 / 10
# Iteration 759 / 759
# Train Loss: 0.0086, Train Accuracy: 0.9521
# Test Loss: 0.0546, Test Accuracy: 0.8392
# Epoch 4 / 10
# Iteration 759 / 759
# Train Loss: 0.0119, Train Accuracy: 0.9501
# Test Loss: 0.0814, Test Accuracy: 0.8036
# Epoch 5 / 10
# Iteration 759 / 759
# Train Loss: 0.0116, Train Accuracy: 0.9521
# Test Loss: 0.0806, Test Accuracy: 0.8090
# Epoch 6 / 10
# Iteration 759 / 759
# Train Loss: 0.0088, Train Accuracy: 0.9597
# Test Loss: 0.0836, Test Accuracy: 0.8232
# Epoch 7 / 10
# Iteration 759 / 759
# Train Loss: 0.0072, Train Accuracy: 0.9630
# Test Loss: 0.0768, Test Accuracy: 0.8249
# Epoch 8 / 10
# Iteration 759 / 759
# Train Loss: 0.0050, Train Accuracy: 0.9752
# Test Loss: 0.0781, Test Accuracy: 0.8281
# Epoch 9 / 10
# Iteration 759 / 759
# Train Loss: 0.0079, Train Accuracy: 0.9652
# Test Loss: 0.0833, Test Accuracy: 0.7934
# Epoch 10 / 10
# Iteration 759 / 759
# Train Loss: 0.0062, Train Accuracy: 0.9651
# Test Loss: 0.0843, Test Accuracy: 0.7917
