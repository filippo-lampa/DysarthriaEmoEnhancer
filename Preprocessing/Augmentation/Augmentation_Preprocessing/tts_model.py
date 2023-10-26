"""Our voice conversion model consists of 6 layers of multihead attention [16]. We implement this model using the
TransformerEncoderLayer from PyTorch where each multihead attention layer has 8 heads and a model dimension of 80
(number of frequency bins). To train the voice conversion model, we pass the time-aligned normal utterance through the
network and apply the mean-squared error (MSE) loss function between the network output and the matching time-aligned
dysarthric utterance. We use the Adam optimizer, a batch-size of one, and train for 150,000 iterations."""

import torch
import torch.nn as nn
import torch.optim as optim


class VoiceConversionTransformer(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, num_frequency_bins):
        super(VoiceConversionTransformer, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, num_heads),
            num_layers
        )
        self.fc = nn.Linear(model_dim, num_frequency_bins)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x


def load_data():
    pass

# Define the mean squared error (MSE) loss
criterion = nn.MSELoss()

# Create an instance of the model
model = VoiceConversionTransformer(model_dim=80, num_heads=8, num_layers=6, num_frequency_bins=80)

# Create an instance of the Adam optimizer
optimizer = optim.Adam(model.parameters())

# Set batch size to 1
batch_size = 1

# Train for 150,000 iterations
num_iterations = 150000

for iteration in range(num_iterations):
    # Load a time-aligned normal utterance and its corresponding dysarthric utterance
    normal_utterance, dysarthric_utterance = load_data()

    # Clear the gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(normal_utterance)

    # Calculate the MSE loss
    loss = criterion(output, dysarthric_utterance)

    # Backpropagation
    loss.backward()

    # Update the model parameters
    optimizer.step()

    if (iteration + 1) % 1000 == 0:
        print(f'Iteration [{iteration + 1}/{num_iterations}] - Loss: {loss.item()}')

# Save the trained model if desired
torch.save(model.state_dict(), 'voice_conversion_model.pt')
