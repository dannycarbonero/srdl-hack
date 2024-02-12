import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# Define the LSTM Network
class SpikeRippleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(SpikeRippleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM layer
        x, (h_n, c_n) = self.lstm(x)

        # Only take the output from the final time step
        x = x[:, -1, :]

        # Normalization layer
        x = self.batch_norm(x)

        # Dropout for regularization
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)

        # Sigmoid activation function
        x = torch.sigmoid(x)

        return x


# Network hyperparameters
input_size = 20  # Because each time point is treated as a feature
hidden_size = 128  # Number of features in the hidden state
num_layers = 2  # Number of recurrent layers
dropout = 0.1  # Dropout rate

# Create the model
model = SpikeRippleLSTM(input_size, hidden_size, num_layers, dropout)

# Loss and Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters())  # Adam optimizer

# Training data
# Assuming `train_data` is your input matrix of shape (200000, 3053) and `train_labels` are your labels
# train_data = torch.randn(200000, 3053, 1)  # Example synthetic data; replace with your actual data
# train_labels = torch.randint(0, 2, (200000, 1)).float()  # Example synthetic labels; replace with your actual labels
data_directory = '/home/warehaus/Neural Data/Spike Ripples/'
spikes = np.array(pd.read_csv(data_directory + 'spikes_0.csv', header = None))
ripples = np.array(pd.read_csv(data_directory + 'ripples_0.csv', header = None))
spikes_labels = np.zeros(spikes.shape[0])
ripples_labels = np.ones(ripples.shape[0])
train_data = torch.tensor(np.vstack((spikes, ripples)), dtype = torch.float)
train_data = torch.unsqueeze(train_data, dim = 2)
train_labels = torch.tensor(np.hstack((spikes_labels, ripples_labels)).reshape(-1,1), dtype = torch.float)

# Convert to PyTorch Dataset
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# Training Loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (sequences, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')


# Training the model
num_epochs = 5  # Number of epochs for training
train(model, train_loader, criterion, optimizer, num_epochs)


# Testing the network on a new sequence
def test_single_sequence(model, sequence):
    model.eval()  # Evaluation mode
    with torch.no_grad():
        sequence = torch.tensor(sequence).view(1, -1, 1).float()  # Reshape and convert to float
        output = model(sequence)
        prediction = 1 if output.item() >= 0.5 else 0