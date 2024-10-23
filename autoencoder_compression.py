import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import os

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, num_hidden):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_hidden = num_hidden

        # Flatten input image dimensions for fully connected layers
        input_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.encoder = nn.Sequential(
            nn.Linear(input_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, latent_dim)
        )

    def forward(self, x):
        x = x.to(x.device)
        x = x.view(x.size(0), -1)  # Flatten the image
        latent_space = self.encoder(x)
        return latent_space

class Decoder(nn.Module):
    def __init__(self, latent_dim, num_hidden, output_shape):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_hidden = num_hidden
        self.output_shape = output_shape

        # Output size is reconstructed to the original image shape
        output_size = output_shape[0] * output_shape[1] * output_shape[2]

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, output_size),
            nn.Sigmoid()  # Ensure output is normalized between 0-1
        )

    def forward(self, x):
        x = x.to(x.device)
        x = self.decoder(x)
        x = x.view(x.size(0), *self.output_shape)  # Reshape back to image
        return x

class NeuralImageCompression:
    def __init__(self, encoder, decoder=None, device="cpu"):
        self.encoder = encoder.to(device) if encoder else None
        self.decoder = decoder.to(device) if decoder else None
        self.device = device

    def compress(self, x):
        x = x.to(self.device)
        return self.encoder(x)

    def decompress(self, z):
        if self.decoder is None:
            raise ValueError("Decoder not available.")
        z = z.to(self.device)
        return self.decoder(z)

    def train_encoder_decoder(self, train_loader, criterion, optimizer, device, epochs=20, regularization=None):
        """ Train both encoder and decoder for task 0 with regularization """
        self.encoder.train()
        self.decoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, _ in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                z = self.encoder(x)
                x_reconstructed = self.decoder(z)
                
                loss = criterion(x_reconstructed, x)

                # Add regularization (L2 penalty)
                if regularization:
                    l2_loss = 0
                    for param in self.encoder.parameters():
                        l2_loss += torch.norm(param)
                    for param in self.decoder.parameters():
                        l2_loss += torch.norm(param)
                    loss += regularization * l2_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def train_encoder(self, train_loader, criterion, optimizer, device, epochs=20, regularization=None, decoder_pth=None):
        """ Train only the encoder for subsequent tasks with fixed decoder. """
        self.encoder.train()

        # Load the saved decoder for task 1 onwards
        self.retrieve_decoder(decoder_pth)
        self.decoder.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for x, _ in train_loader:
                x = x.to(device)
                optimizer.zero_grad()

                # Forward pass
                z = self.encoder(x)
                x_reconstructed = self.decoder(z)
                
                loss = criterion(x_reconstructed, x)

                # Add regularization (L2 penalty)
                if regularization:
                    l2_loss = 0
                    for param in self.encoder.parameters():
                        l2_loss += torch.norm(param)
                    loss += regularization * l2_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def retrieve_decoder(self, decoder_pth):
        """ Retrieve the saved decoder from the file """
        if decoder_pth:
            self.decoder = torch.load(decoder_pth, map_location=self.device)
            self.decoder.eval()
        else:
            raise ValueError("Decoder path must be provided to load the model.")
    
    def get_decoder_latent_dim(self):
        """ Function to extract the latent dimension from the saved decoder """
        if self.decoder is None:
            raise ValueError("Decoder is not initialized. Please load a decoder first.")
        
        # Assuming the decoder's first layer is Linear with input size = latent dimension
        first_layer = self.decoder.decoder[0]  # Access the first layer in the Sequential model
        if isinstance(first_layer, nn.Linear):
            return first_layer.in_features
        else:
            raise ValueError("First layer of the decoder is not a Linear layer.")
        
    @staticmethod
    def hyperparameter_tuning(train_loader, task_id, device, latent_dim, num_hidden, criterion, regularization=None):
        """ Use Optuna to tune hyperparameters for task 0, and encoder only for task 1 onwards. """

        def objective(trial):
            # Task 0: Tune both encoder and decoder
            if task_id == 0:
                latent_dim = trial.suggest_int("latent_dim", 16, 128, step=16)
                num_hidden = trial.suggest_int("num_hidden", 64, 256, step=64)
                regularization = trial.suggest_float("regularization", 1e-5, 1e-3, log=True)
                learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

                # Define encoder and decoder
                encoder = Encoder(input_shape=(1, 28, 28), latent_dim=latent_dim, num_hidden=num_hidden).to(device)
                decoder = Decoder(latent_dim=latent_dim, num_hidden=num_hidden, output_shape=(1, 28, 28)).to(device)

                # Train both
                nic = NeuralImageCompression(encoder, decoder, device=device)
                optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
                nic.train_encoder_decoder(train_loader, criterion, optimizer, device, regularization=regularization)

            # Task 1 onwards: Tune encoder only
            else:
                nic = NeuralImageCompression(encoder=None, device=device)
                path = "best_decoder_task_0.pth"
                nic.retrieve_decoder(decoder_pth=path)
                expected_latent_dim = nic.get_decoder_latent_dim()
                print("Expected Latent Dimension:", expected_latent_dim)

                # Now, continue tuning the encoder with the same latent_dim
                num_hidden = trial.suggest_int("num_hidden", 64, 256, step=64)
                regularization = trial.suggest_float("regularization", 1e-5, 1e-3, log=True)
                learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

                encoder = Encoder(input_shape=(1, 28, 28), latent_dim=expected_latent_dim, num_hidden=num_hidden).to(device)
                optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
                nic.encoder = encoder  # Set the encoder in the NeuralImageCompression instance
                nic.decoder = nic.decoder.to(device)  # Ensure decoder is on the same device
                nic.train_encoder(train_loader, criterion, optimizer, device, regularization=regularization, decoder_pth=path)

            # Here we compute loss over validation set or another objective metric
            total_loss = 0
            with torch.no_grad():
                for x, _ in train_loader:
                    x = x.to(device)
                    nic.encoder = nic.encoder.to(device)
                    z = nic.compress(x)
                    x_reconstructed = nic.decompress(z)
                    loss = criterion(x_reconstructed, x)
                    total_loss += loss.item()
            return total_loss / len(train_loader)  # Return the average loss for Optuna

        # Run Optuna for hyperparameter tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        # Return best parameters from Optuna
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
        return best_params