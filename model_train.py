import torch
from dataloader import BuildData, DataBuilder
from model import MLPVGG163DAdjusted
import json
import torch.optim as optim


# Prepare Data

DATA_PATH = 'data_processed.csv'
SPLIT_RATIO = 0.6
databuilder = DataBuilder(DATA_PATH, SPLIT_RATIO)
datasets = databuilder.prepare()
print(f"Shape of training data: {len(datasets['train'])}")
print(f"Shape of validation data: {len(datasets['validate'])}")

# Make data tensors

BATCH_SIZE = 4
trainloader = torch.utils.data.DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(datasets['validate'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print("dataloader successful!")

# Define Model

with open('vgg16_adj_configs.json') as config_file:
        model_configs = json.load(config_file)
model_save_path = "./model_weights/"    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is set to: {device}")

model = MLPVGG163DAdjusted(model_configs=model_configs, num_classes=2, device=device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train Model 

EPOCHS = 10

model.train(True)
for idx, (image, features, label) in enumerate(trainloader):
        image = image.to(device)
        features = features.to(device)
        label = label.to(device)

        output = model(image, features)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"For {idx}/{len(trainloader)} Loss: {loss.item()}")

predicted = []
total = 0
correct = 0
model.eval()
with torch.no_grad():
        for idx, (image, features, label) in enumerate(valloader):
                image = image.to(device)
                features = features.to(device)
                label = label.to(device)
                output = model(image, features)
                _, pred = torch.max(output.data, 1)
                predicted.append(pred)

                total += label.size(0)
                correct += (label == pred).sum().item()

        accuracy = correct/total

print(f"Accuracy: {accuracy}")



