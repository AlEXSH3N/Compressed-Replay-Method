import cupy as cp
import numpy as np
from sklearn.decomposition import PCA
import torch
import optuna
import joblib 
import os

class PCACompression:
    def __init__(self, n_components, whiten, svd_solver, device="cpu"):
        self.n_components = n_components
        self.device = device
        self.is_gpu = device == "cuda" and cp.cuda.is_available()
        self.pca = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver)

    def fit(self, data):
        flat_data = data.view(data.size(0), -1)
        if self.is_gpu:
            flat_data = cp.asarray(flat_data.cpu().numpy())  # Move to GPU
        else:
            flat_data = flat_data.cpu().numpy()

        self.pca.fit(flat_data)

    def compress(self, data):
        flat_data = data.view(data.size(0), -1)
        if self.is_gpu:
            flat_data = cp.asarray(flat_data.cpu().numpy())  # Move to GPU
            compressed = self.pca.transform(flat_data.get())  # Use CPU for sklearn
            return torch.tensor(compressed, device=self.device)
        else:
            compressed = self.pca.transform(flat_data.cpu().numpy())
            return torch.tensor(compressed)

    def decompress(self, compressed_data, shape):
        if self.is_gpu:
            decompressed = self.pca.inverse_transform(compressed_data.cpu().numpy())
            decompressed = cp.asarray(decompressed)  # Move back to GPU
        else:
            decompressed = self.pca.inverse_transform(compressed_data.cpu().numpy())

        return torch.tensor(decompressed).view(-1, *shape)

    @staticmethod
    def tune_hyperparameters(data, input_shape, device="cpu"):
        def objective(trial):
            low_bound = max(40, int(input_shape[1] * input_shape[2] * 0.15))
            high_bound = min(int(input_shape[1] * input_shape[2] * 0.4), 96)

            # Ensure low_bound does not exceed high_bound
            if low_bound > high_bound:
                low_bound, high_bound = high_bound, low_bound
            n_components = trial.suggest_int("n_components", low_bound, high_bound)
            whiten = trial.suggest_categorical("whiten", [False])
            svd_solver = trial.suggest_categorical("svd_solver", ["auto"])
            pca = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver)

            flat_data = data.view(data.size(0), -1)
            if device == "cuda" and cp.cuda.is_available():
                flat_data = cp.asarray(flat_data.cpu().numpy())
            else:
                flat_data = flat_data.cpu().numpy()

            pca.fit(flat_data)
            compressed = pca.transform(flat_data)
            decompressed = pca.inverse_transform(compressed)
            reconstruction_error = np.mean((flat_data - decompressed) ** 2)
            return reconstruction_error

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        print("Best PCA n_components:", best_params["n_components"], "Whiten:", best_params["whiten"], "SVD Solver:", best_params["svd_solver"])
        return best_params["n_components"], best_params["whiten"], best_params["svd_solver"]

    def save_pca_params(self, task_id):
        path = f"pca_compressed_metadata/pca_params_task_{task_id}.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pca_params = {"components": self.pca.components_, "mean": self.pca.mean_}
        joblib.dump(pca_params, path)

    @staticmethod
    def load_pca_params(path="pca_params.pkl", n_components=20):
        pca_params = joblib.load(path)
        pca = PCA(n_components=n_components)
        pca.components_ = pca_params["components"]
        pca.mean_ = pca_params["mean"]
        return pca

'''
More than 1 channel PCA compression below
'''

# class PCACompression:
#     def __init__(self, n_components, whiten, svd_solver, device="cpu"):
#         self.n_components = n_components
#         self.device = device
#         self.is_gpu = device == "cuda" and cp.cuda.is_available()
#         self.pca = [PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver) for _ in range(3)]

#     def fit(self, data):
#         # Separate data into three channels and flatten each
#         channels = [data[:, c, :, :].view(data.size(0), -1) for c in range(3)]
#         for i in range(3):
#             flat_data = channels[i]
#             if self.is_gpu:
#                 flat_data = cp.asarray(flat_data.cpu().numpy())  # Move to GPU
#             else:
#                 flat_data = flat_data.cpu().numpy()
#             self.pca[i].fit(flat_data)

#     def compress(self, data):
#         compressed_channels = []
#         for i in range(3):
#             flat_data = data[:, i, :, :].view(data.size(0), -1)
#             if self.is_gpu:
#                 flat_data = cp.asarray(flat_data.cpu().numpy())  # Move to GPU
#                 compressed = self.pca[i].transform(flat_data.get())  # Use CPU for sklearn
#             else:
#                 compressed = self.pca[i].transform(flat_data.cpu().numpy())
#             compressed_channels.append(torch.tensor(compressed, device=self.device))
        
#         return torch.stack(compressed_channels, dim=1)  # Shape: (num_samples, 3, n_components)

#     def decompress(self, compressed_data, shape=(3, 32, 32)):
#         num_images = compressed_data.shape[0]
#         reconstructed_images = []

#         for i in range(num_images):
#             image_reconstructed_channels = []
#             for c in range(3):  # Loop over each channel
#                 compressed_channel = compressed_data[i, c, :]
#                 if self.is_gpu:
#                     decompressed = self.pca[c].inverse_transform(compressed_channel.cpu().numpy())
#                     decompressed = cp.asarray(decompressed)  # Move back to GPU if necessary
#                 else:
#                     decompressed = self.pca[c].inverse_transform(compressed_channel.cpu().numpy())

#                 # Reshape to (H, W) and add to image_reconstructed_channels
#                 image_reconstructed_channels.append(torch.tensor(decompressed).view(shape[1:]))
            
#             # Stack the three channels together as (3, H, W) for each image
#             reconstructed_image = torch.stack(image_reconstructed_channels, dim=0)
#             reconstructed_images.append(reconstructed_image)

#         # Convert to tensor with shape (N, 3, H, W)
#         return torch.stack(reconstructed_images, dim=0)  # Shape: (num_images, 3, H, W)

#     @staticmethod
#     def tune_hyperparameters(data, input_shape, device="cpu"):
#         def objective(trial):
#             low_bound, high_bound = 50, 100
#             n_components = trial.suggest_int("n_components", low_bound, high_bound)
#             whiten = trial.suggest_categorical("whiten", [False])
#             svd_solver = trial.suggest_categorical("svd_solver", ["auto"])
#             pca = [PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver) for _ in range(3)]

#             channels = [data[:, c, :, :].view(data.size(0), -1) for c in range(3)]
#             reconstruction_errors = []
#             for i in range(3):
#                 flat_data = channels[i]
#                 if device == "cuda" and cp.cuda.is_available():
#                     flat_data = cp.asarray(flat_data.cpu().numpy())
#                 else:
#                     flat_data = flat_data.cpu().numpy()
#                 pca[i].fit(flat_data)
#                 compressed = pca[i].transform(flat_data)
#                 decompressed = pca[i].inverse_transform(compressed)
#                 reconstruction_error = np.mean((flat_data - decompressed) ** 2)
#                 reconstruction_errors.append(reconstruction_error)
#             return np.mean(reconstruction_errors)

#         study = optuna.create_study(direction="minimize")
#         study.optimize(objective, n_trials=50)
#         best_params = study.best_params
#         print("Best PCA n_components:", best_params["n_components"], "Whiten:", best_params["whiten"], "SVD Solver:", best_params["svd_solver"])
#         return best_params["n_components"], best_params["whiten"], best_params["svd_solver"]

#     def save_pca_params(self, task_id):
#         os.makedirs("pca_compressed_metadata", exist_ok=True)
#         for i in range(3):
#             path = f"pca_compressed_metadata/pca_params_task_{task_id}_channel_{i}.pkl"
#             pca_params = {"components": self.pca[i].components_, "mean": self.pca[i].mean_}
#             joblib.dump(pca_params, path)

#     @staticmethod
#     def load_pca_params(path="pca_params.pkl", n_components=20):
#         pca = [PCA(n_components=n_components) for _ in range(3)]
#         for i in range(3):
#             channel_path = path.replace(".pkl", f"_channel_{i}.pkl")
#             pca_params = joblib.load(channel_path)
#             pca[i].components_ = pca_params["components"]
#             pca[i].mean_ = pca_params["mean"]
#         return pca
