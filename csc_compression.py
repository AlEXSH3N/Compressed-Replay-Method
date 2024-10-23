import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
import scipy.sparse as sp
import cupy as cp  # Ensure that cupy is installed for CUDA-based sparse operations if needed.
from torch.utils.data import DataLoader, TensorDataset

class ConvolutionalSparseCoding:
    def __init__(self, num_filters, kernel_size, sparsity_weight=1e-4, device="cpu"):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sparsity_weight = sparsity_weight
        self.device = device
        self.is_cuda = device == "cuda" and torch.cuda.is_available()
        self._build_model()
        
    def _build_model(self):
        """Build the convolutional dictionary layer."""
        self.conv_dict = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=False
        ).to(self.device)

    def forward(self, x):
        """Forward pass to get sparse codes and reconstruction."""
        sparse_codes = self.conv_dict(x)  # Sparse encoding
        l1_penalty = self.sparsity_weight * torch.norm(sparse_codes, p=1)
        
        reconstruction = F.conv_transpose2d(
            sparse_codes, 
            self.conv_dict.weight,
            stride=1,
            padding=self.conv_dict.kernel_size[0] // 2
        )
        return reconstruction, l1_penalty

    def train(self, train_loader, num_epochs=10, learning_rate=1e-3):
        """Train the CSC model with reconstruction loss and L1 penalty."""
        optimizer = torch.optim.Adam(self.conv_dict.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for images, _ in train_loader:
                images = images.to(self.device)

                # Forward pass
                reconstruction, l1_penalty = self.forward(images)
                mse_loss = F.mse_loss(reconstruction, images)
                loss = mse_loss + l1_penalty

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    def compress(self, data):
        """Compress the data to sparse codes in sparse matrix format."""
        with torch.no_grad():
            sparse_codes = self.conv_dict(data.to(self.device))
            sparse_codes_np = sparse_codes.cpu().numpy().reshape(sparse_codes.size(0), -1)

            # Convert to sparse matrix format (using scipy)
            compressed_sparse_codes = [
                sp.csr_matrix(sparse_code) for sparse_code in sparse_codes_np
            ]
        return compressed_sparse_codes

    def decompress(self, compressed_sparse_codes, input_shape=None):
        """Decompress sparse codes back to the original data format."""
        decompressed_data = []
        for sparse_code in compressed_sparse_codes:
            # Convert back to dense format
            if isinstance(sparse_code, sp.csr_matrix):
                dense_code = torch.tensor(sparse_code.toarray(), device=self.device).view(1, self.num_filters, *input_shape[1:])
            else:
                dense_code = sparse_code.view(*input_shape[1:])
            
            # Perform transposed convolution to reconstruct the original image
            reconstructed = F.conv_transpose2d(
                dense_code, 
                self.conv_dict.weight, 
                stride=1,
                padding=self.conv_dict.kernel_size[0] // 2
            )
            reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-5)
            decompressed_data.append(reconstructed)
            # .view(*input_shape)
        return torch.cat(decompressed_data)

    @staticmethod
    def tune_hyperparameters(data, input_shape, device="cpu"):
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            # Hyperparameters to tune
            num_filters = trial.suggest_int("num_filters", 8, 32, step=4)
            kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
            sparsity_weight = trial.suggest_float("sparsity_weight", 5e-5, 5e-4, log=True)

            # Instantiate model with trial hyperparameters
            model = ConvolutionalSparseCoding(
                num_filters=num_filters,
                kernel_size=kernel_size,
                sparsity_weight=sparsity_weight,
                device=device
            )

            # Fit model on data (one epoch as sample)
            model.train(train_loader=data, num_epochs=1, learning_rate=1e-3)

            # Compress and decompress sample batch
            images, _ = next(iter(data))
            images = images.to(device)
            sparse_codes = model.compress(images)
            reconstructed = model.decompress(sparse_codes, input_shape)

            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(reconstructed, images).item()
            return reconstruction_error

        # Create and optimize study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        
        best_params = study.best_params
        print("Best CSC hyperparameters:", best_params)
        return best_params

'''
More than 1 channel CSC model below
'''
# class ConvolutionalSparseCoding:
#     def __init__(self, num_filters, kernel_size, sparsity_weight=1e-4, device="cpu"):
#         self.num_filters = num_filters
#         self.kernel_size = kernel_size
#         self.sparsity_weight = sparsity_weight
#         self.device = device
#         self.is_cuda = device == "cuda" and torch.cuda.is_available()
#         self._build_model()
        
#     def _build_model(self):
#         """Build the convolutional dictionary layer."""
#         self.conv_dict = nn.Conv2d(
#             in_channels=3,  # For RGB images
#             out_channels=self.num_filters,
#             kernel_size=self.kernel_size,
#             stride=1,
#             padding=self.kernel_size // 2,
#             bias=False
#         ).to(self.device)

#     def forward(self, x):
#         """Forward pass to get sparse codes and reconstruction."""
#         sparse_codes = self.conv_dict(x)  # Sparse encoding
#         l1_penalty = self.sparsity_weight * torch.norm(sparse_codes, p=1)
        
#         reconstruction = F.conv_transpose2d(
#             sparse_codes, 
#             self.conv_dict.weight,
#             stride=1,
#             padding=self.conv_dict.kernel_size[0] // 2
#         )
#         return reconstruction, l1_penalty

#     def train(self, image_loader, num_epochs=10, learning_rate=1e-3):
#         """Train the CSC model with reconstruction loss and L1 penalty."""
#         optimizer = torch.optim.Adam(self.conv_dict.parameters(), lr=learning_rate)
#         for epoch in range(num_epochs):
#             epoch_loss = 0
#             for images in image_loader:
#                 images = images.to(self.device)

#                 # Forward pass
#                 reconstruction, l1_penalty = self.forward(images)
#                 mse_loss = F.mse_loss(reconstruction, images)
#                 loss = mse_loss + l1_penalty

#                 # Backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item()

#             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(image_loader):.4f}")

#     def compress(self, data):
#         """Compress the data to sparse codes in sparse matrix format."""
#         with torch.no_grad():
#             sparse_codes = self.conv_dict(data.to(self.device))
#             sparse_codes_np = sparse_codes.cpu().numpy().reshape(sparse_codes.size(0), -1)

#             # Convert to sparse matrix format
#             compressed_sparse_codes = [
#                 sp.csr_matrix(sparse_code) for sparse_code in sparse_codes_np
#             ]
#         return compressed_sparse_codes

#     def decompress(self, compressed_sparse_codes, input_shape=None):
#         """Decompress sparse codes back to the original data format."""
#         decompressed_data = []
#         for sparse_code in compressed_sparse_codes:
#             sparse_code = sparse_code[0]
#             # Check if sparse_code is a sparse matrix and convert if necessary
#             if isinstance(sparse_code, sp.csr_matrix):
#                 dense_code = torch.tensor(sparse_code.toarray(), device=self.device).view(1, self.num_filters, *input_shape[1:])
#             else:
#                 dense_code = sparse_code.view(1, self.num_filters, *input_shape[1:])
            
#             # Perform transposed convolution to reconstruct the original image
#             reconstructed = F.conv_transpose2d(
#                 dense_code, 
#                 self.conv_dict.weight, 
#                 stride=1,
#                 padding=self.conv_dict.kernel_size[0] // 2
#             )
#             # Normalize reconstructed values
#             reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-5)
#             decompressed_data.append(reconstructed)
        
#         # Combine the decompressed images and reshape to RGB format
#         print(len(decompressed_data))
#         return torch.cat(decompressed_data, dim=0).view(-1, 3, *input_shape[1:])


#     @staticmethod
#     def tune_hyperparameters(image_loader, input_shape, device="cpu"):
#         """Optimize hyperparameters using Optuna."""
#         def objective(trial):
#             # Hyperparameters to tune
#             num_filters = trial.suggest_int("num_filters", 8, 32, step=4)
#             kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
#             sparsity_weight = trial.suggest_float("sparsity_weight", 5e-5, 5e-4, log=True)

#             # Instantiate model with trial hyperparameters
#             model = ConvolutionalSparseCoding(
#                 num_filters=num_filters,
#                 kernel_size=kernel_size,
#                 sparsity_weight=sparsity_weight,
#                 device=device
#             )

#             # Train model briefly on the image_loader
#             model.train(image_loader=image_loader, num_epochs=1, learning_rate=1e-3)

#             # Compress and decompress sample batch
#             images = next(iter(image_loader)).to(device)
#             sparse_codes = model.compress(images)
#             reconstructed = model.decompress(sparse_codes, input_shape)

#             # Calculate reconstruction error
#             reconstruction_error = F.mse_loss(reconstructed, images).item()
#             return reconstruction_error

#         # Optimize study
#         study = optuna.create_study(direction="minimize")
#         study.optimize(objective, n_trials=30)
        
#         best_params = study.best_params
#         print("Best CSC hyperparameters:", best_params)
#         return best_params
