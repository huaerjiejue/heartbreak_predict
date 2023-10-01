import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from torch.utils.tensorboard import SummaryWriter


# Define the Neural Network model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(10, 5),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(5, 1),
            nn.Sigmoid(),  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.layers(x)


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Load and preprocess data
data = pd.read_csv("../data/heart.csv")
X_data = data.drop(["target"], axis=1)
y = data["target"]
x_train, x_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.values)  # 不添加变量名称防止后面会出现没有名称的警告
x_test = scaler.transform(x_test)

# Convert data to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train)
y_train_tensor = torch.FloatTensor(y_train.values)
x_test_tensor = torch.FloatTensor(x_test)
y_test_tensor = torch.FloatTensor(y_test.values)

# Create DataLoader for training and testing
train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = BinaryClassifier(input_dim=x_train.shape[1])
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.02)

# logdir = "../src/logs"
# writer = SummaryWriter(logdir)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(
            y_pred, y_batch.unsqueeze(1)
        )  # BCELoss expects target in (batch_size, 1) shape
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    # writer.add_scalar("Loss/train", loss.item(), epoch + 1)  # 记录训练过程的损失

# writer.close()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(x_train_tensor)
    y_pred_class = (y_pred >= 0.5).float()  # Threshold at 0.5 for binary classification
    train_accuracy = (y_pred_class == y_train_tensor.unsqueeze(1)).float().mean()
    print(f"Train Accuracy: {train_accuracy.item()}")
    y_pred = model(x_test_tensor)
    y_pred_class = (y_pred >= 0.5).float()  # Threshold at 0.5 for binary classification
    test_accuracy = (y_pred_class == y_test_tensor.unsqueeze(1)).float().mean()
    print(f"Test Accuracy: {test_accuracy.item()}")


save_model = input("Do you want to model the model? (Y/n): ")
if save_model.lower() == "y" or save_model == "":
    model_path = f"../model/nn_model.pt"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, "../model/scaler.pkl")
    joblib.dump(train_accuracy.item(), "../model/train_accuracy.pkl")
    joblib.dump(test_accuracy.item(), "../model/test_accuracy.pkl")
    print(f"Model saved as {model_path}")
