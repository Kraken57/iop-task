# **SPWM Dataset Generation and Prediction Using Machine Learning**

## **Introduction**

This project involves generating a dataset of Sinusoidal Pulse Width Modulation (SPWM) signals using MATLAB and then training a neural network in Python (PyTorch) to predict the SPWM waveform given a modulation index. The system ensures that two switches are never ON simultaneously, making it suitable for half-bridge inverters.

## **Process Overview**

1. **Generate the SPWM dataset using MATLAB**:
   - Create sinusoidal reference signals.
   - Generate a high-frequency carrier wave (triangular wave).
   - Compare the sine wave with the carrier wave to produce an SPWM waveform.
   - Save the dataset as a CSV file.

2. **Train a Neural Network using PyTorch**:
   - Load the generated dataset.
   - Prepare input-output pairs.
   - Define and train a feedforward neural network.
   - Predict and visualize the SPWM waveform for a given modulation index.

---

## **1. MATLAB Code: SPWM Dataset Generation**

### **Explanation**

The MATLAB script generates a dataset containing SPWM waveforms corresponding to different modulation indices. It follows these steps:

### **Steps in MATLAB Code**

1. **Define Parameters:**
   - `num_samples = 5000`: Number of SPWM waveforms to generate.
   - `seq_length = 1000`: Number of time steps per waveform.
   - `switching_freq = 60 kHz`: Carrier wave frequency.
   - `fundamental_freq = 50 Hz`: Base sinusoidal frequency.

2. **Generate Modulation Indices:**
   - `modulation_indices = rand(num_samples, 1) * 1.2;`
   - Random values between `0` and `1.2` are generated.

3. **Create SPWM Waveforms:**
   - Generate a sinusoidal reference signal: `sin_wave = M * sin(2 * pi * fundamental_freq * t);`
   - Generate a triangular carrier wave: `carrier_wave = 2 * abs(mod(switching_freq * t, 1) - 0.5);`
   - Compare the two signals: `spwm_wave = double(sin_wave > carrier_wave);`

4. **Save Dataset:**
   - The dataset is saved as `spwm_dataset_60kHz.csv`.
   - The first column stores the modulation index, and the remaining columns store the SPWM waveform.

5. **Plot Example SPWM Waveforms:**
   - Two random waveforms are plotted for visualization.

```matlab
% Define parameters
num_samples = 5000;
seq_length = 1000;
switching_freq = 60e3;
fundamental_freq = 50;
modulation_indices = rand(num_samples, 1) * 1.2;
spwm_data = zeros(num_samples, seq_length);
t = linspace(0, 1/fundamental_freq, seq_length);

for i = 1:num_samples
    M = modulation_indices(i);
    sin_wave = M * sin(2 * pi * fundamental_freq * t);
    carrier_wave = 2 * abs(mod(switching_freq * t, 1) - 0.5);
    spwm_wave = double(sin_wave > carrier_wave);
    spwm_data(i, :) = spwm_wave;
end

csvwrite('spwm_dataset_60kHz.csv', [modulation_indices, spwm_data]);
```

---

## **2. PyTorch Code: Neural Network for SPWM Prediction**

### **Explanation**

The Python script loads the generated dataset, trains a neural network using PyTorch, and predicts the SPWM waveform given a modulation index.

### **Steps in Python Code**

1. **Load the Dataset:**
   - `data = pd.read_csv("spwm_dataset_60kHz.csv", header=None).values`
   - The first column contains the modulation index.
   - The remaining columns store the SPWM waveform.

2. **Prepare Data:**
   - Convert inputs to PyTorch tensors.
   - Use `DataLoader` for batch processing.

3. **Define Neural Network:**
   - A three-layer fully connected network.
   - Uses ReLU activation for hidden layers and Sigmoid for output.

4. **Train the Model:**
   - Uses Binary Cross Entropy Loss.
   - Optimized using Adam optimizer.
   - Runs for 500 epochs.

5. **Test the Model:**
   - A modulation index (e.g., `0.8`) is provided.
   - The trained model generates the corresponding SPWM waveform.
   - The output is plotted.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("spwm_dataset_60kHz.csv", header=None).values
modulation_indices = data[:, 0].astype(np.float32)
spwm_waves = data[:, 1:].astype(np.float32)

# Convert to PyTorch tensors
inputs = torch.tensor(modulation_indices).view(-1, 1)
targets = torch.tensor(spwm_waves)

# DataLoader
train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=64, shuffle=True)

# Define Neural Network
class SPWMNet(nn.Module):
    def __init__(self, output_size):
        super(SPWMNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Model Initialization
output_size = spwm_waves.shape[1]
spwm_model = SPWMNet(output_size).to("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCELoss()
optimizer = optim.Adam(spwm_model.parameters(), lr=0.0005)

# Training Function
def train_model(model, dataloader, epochs=500):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.6f}")

train_model(spwm_model, train_loader)

# Test Model
spwm_model.eval()
test_input = torch.tensor([[0.8]], dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
predicted_wave = spwm_model(test_input).detach().cpu().numpy().flatten()

# Plot Result
plt.plot(predicted_wave[:500], label="Predicted SPWM Wave")
plt.xlabel("Time Step")
plt.ylabel("SPWM Output")
plt.title("Generated SPWM Wave for Modulation Index 0.8")
plt.legend()
plt.grid()
plt.show()
```

---

## **Conclusion**

This workflow demonstrates how to generate an SPWM dataset in MATLAB and use a neural network in PyTorch to predict SPWM waveforms given a modulation index. The trained model successfully maps modulation indices to their respective SPWM waveforms.

