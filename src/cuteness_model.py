# cuteness_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

class CutenessModel(nn.Module):
    def __init__(self):
        super(CutenessModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_cuteness_model(data_dir, num_epochs=10):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = ImageFolder(data_dir, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = CutenessModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Validation dataset (you can create a separate validation dataset if needed)
    validation_dataset = ImageFolder("data/validation", transform=data_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

    with open('training_log.txt', 'w') as log_file:
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.float().view(-1, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print(f'Training Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')
                    log_file.write(f'Training Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}\n')

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_labels in validation_loader:
                    val_outputs = model(val_inputs)
                    val_labels = val_labels.float().view(-1, 1)
                    val_loss += criterion(val_outputs, val_labels).item()

            val_loss /= len(validation_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}')
            log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}\n')

    # Save the trained model
    torch.save(model.state_dict(), 'cuteness_model.pth')

if __name__ == "__main__":
    data_directory = "data/train"  # Update with your actual data directory
    train_cuteness_model(data_directory)
