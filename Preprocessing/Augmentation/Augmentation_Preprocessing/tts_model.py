import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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


def train(normal_utterances, dysarthric_utterances):
    # Define the mean squared error (MSE) loss
    criterion = nn.MSELoss()

    # Create an instance of the model
    model = VoiceConversionTransformer(model_dim=80, num_heads=8, num_layers=6, num_frequency_bins=80)

    # Create an instance of the Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Train for 150,000 iterations
    num_iterations = 150000

    n_utterance = ""
    d_utterance = ""

    for iteration in range(num_iterations):
        # Load a time-aligned normal utterance and its corresponding dysarthric utterance

        if len(normal_utterances) > 0 and len(dysarthric_utterances) > 0:
            n_utterance = np.array(normal_utterances.pop(0), dtype=np.float32)
            d_utterance = np.array(dysarthric_utterances.pop(0), dtype=np.float32)
        else:
            print("No more normal and dysarthric utterances")
            break

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(n_utterance)

        # Calculate the MSE loss
        loss = criterion(output, d_utterance)

        # Backpropagation
        loss.backward()

        # Update the model parameters
        optimizer.step()

        if (iteration + 1) % 1000 == 0:
            print(f'Iteration [{iteration + 1}/{num_iterations}] - Loss: {loss.item()}')

    # Save the trained model if desired
    torch.save(model.state_dict(), 'voice_conversion_model.pt')
