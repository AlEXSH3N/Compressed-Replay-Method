from skopt import gp_minimize
from skopt.space import Integer
from sklearn.decomposition import DictionaryLearning
import numpy as np

from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import Optimizer
from sklearn.decomposition import DictionaryLearning
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import cupy as cp

def optimize_ksvd(flattened_images, n_calls=20):
    # Define the search space for K-SVD parameters
    search_space = [
        Integer(50, 90, name='num_atoms'),             # Range for num_atoms
        Integer(15, 30, name='sparsity_level')           # Range for sparsity_level
    ]
    
    # Initialize optimizer
    optimizer = Optimizer(search_space, random_state=42)
    
    # Objective function for each parameter set
    def objective(params):
        num_atoms, sparsity_level = params
        dict_learner = DictionaryLearning(
            n_components=num_atoms,
            transform_algorithm='omp',
            transform_n_nonzero_coefs=sparsity_level,
            max_iter=20
        )
        
        # Fit dictionary learning and calculate reconstruction error
        sparse_codes = dict_learner.fit_transform(flattened_images)
        reconstruction_error = np.mean((flattened_images - np.dot(sparse_codes, dict_learner.components_)) ** 2)
        return reconstruction_error  # Minimize reconstruction error
        # flattened_images = torch.tensor(flattened_images, dtype=torch.float32).cuda()
        # input_size = flattened_images.shape[1]
        # dict_learner = CustomDictionaryLearning(num_atoms, input_size, sparsity_level).cuda()
        # sparse_codes, dictionary, final_loss = dict_learner.fit(flattened_images)
        # return final_loss
        
    
    # Run Bayesian optimization with tqdm progress bar
    best_score = float('inf')
    best_params = None
    with tqdm(total=n_calls, desc="Optimizing K-SVD") as pbar:
        for _ in range(n_calls):
            suggested_params = optimizer.ask()
            score = objective(suggested_params)
            optimizer.tell(suggested_params, score)
            
            # Update best parameters if current score is lower
            if score < best_score:
                best_score = score
                best_params = suggested_params
            
            pbar.update(1)
    
    best_num_atoms, best_sparsity_level = best_params
    
    # Get the final best sparse codes and dictionary
    dict_learner = DictionaryLearning(
        n_components=best_num_atoms,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=best_sparsity_level,
        max_iter=20
    )
    best_sparse_codes = dict_learner.fit_transform(flattened_images)
    best_dictionary = dict_learner.components_

    final_params = {
        'num_atoms': best_num_atoms,
        'sparsity_level': best_sparsity_level
    }
    print(f"Best K-SVD Params: {final_params}")
    
    return best_sparse_codes, best_dictionary, final_params


class KSVDDictionary:
    def __init__(self, dictionary, image_shape):
        self.dictionary = dictionary  # Learned dictionary from optimize_ksvd
        self.image_shape = image_shape  # Original shape of the images for reconstruction
    
    def reconstruct(self, sparse_codes):
        if isinstance(sparse_codes, torch.Tensor):
            sparse_codes = sparse_codes.detach().cpu().numpy()
        
        if sparse_codes.shape[1] != self.dictionary.shape[0]:  # Check for alignment
            raise ValueError(f"Shape mismatch: sparse_codes {sparse_codes.shape} and dictionary {self.dictionary.shape}")
        
        reconstructed_flat = np.dot(sparse_codes, self.dictionary)
        reconstructed_images = reconstructed_flat.reshape((-1, *self.image_shape))
        return torch.tensor(reconstructed_images, dtype=torch.float32)

"""
Borderline
"""


# def gpu_ksvd(flattened_images, num_atoms=100, sparsity_level=10, max_iter=30):
#     # Convert images to cupy arrays for GPU operations
#     flattened_images = cp.array(flattened_images)
#     num_samples, num_features = flattened_images.shape

#     # Initialize dictionary with random atoms on GPU
#     dictionary = cp.random.randn(num_features, num_atoms)
#     dictionary = cp.linalg.qr(dictionary)[0]  # Orthogonalize initial dictionary

#     for _ in range(max_iter):
#         # Step 1: Sparse Coding - solve for sparse codes using a simplified method or OMP if available
#         sparse_codes = orthogonal_matching_pursuit(flattened_images, dictionary, sparsity_level)

#         # Step 2: Dictionary Update
#         for j in range(num_atoms):
#             indices = cp.nonzero(sparse_codes[:, j])[0]
#             if len(indices) == 0:
#                 continue

#             # Calculate the residual for the selected indices
#             residual = flattened_images[indices] - sparse_codes[indices] @ dictionary.T
#             residual += cp.outer(sparse_codes[indices, j], dictionary[:, j])

#             # Singular Value Decomposition for dictionary update
#             u, s, v = cp.linalg.svd(residual, full_matrices=False)

#             # Ensure u[:, 0] matches the shape of dictionary[:, j]
#             dictionary[:, j] = u[:, 0].reshape(dictionary[:, j].shape)

#             # Assign the updated sparse codes for the selected indices
#             sparse_codes[indices, j] = s[0] * v[0, 0]



#     # Calculate reconstruction error
#     reconstruction_error = cp.mean((flattened_images - sparse_codes @ dictionary.T) ** 2).get()

#     return sparse_codes, dictionary, reconstruction_error

# def orthogonal_matching_pursuit(X, D, sparsity_level):
#     # Convert X and D to cupy arrays if they aren't already
#     X = cp.array(X)
#     D = cp.array(D)

#     # Placeholder sparse codes matrix
#     sparse_codes = cp.zeros((X.shape[0], D.shape[1]))

#     # OMP process for each sample
#     for i in range(X.shape[0]):
#         residual = X[i]
#         indices = []
        
#         for _ in range(sparsity_level):
#             # Find the atom that best matches the residual
#             correlations = cp.abs(D.T @ residual)
#             best_index = cp.argmax(correlations)
#             indices.append(best_index)

#             # Update sparse code for the selected atom
#             selected_atoms = D[:, indices]
#             coeffs, _, _, _ = cp.linalg.lstsq(selected_atoms, X[i])
#             sparse_codes[i, indices] = coeffs

#             # Update the residual
#             residual = X[i] - selected_atoms @ coeffs

#             # Break if the residual is close to zero
#             if cp.linalg.norm(residual) < 1e-6:
#                 break

#     return sparse_codes

# def optimize_ksvd(flattened_images, n_calls=20):
#     # Define the search space for K-SVD parameters
#     search_space = [
#         Integer(50, 200, name='num_atoms'),
#         Integer(5, 20, name='sparsity_level')
#     ]

#     optimizer = Optimizer(search_space, random_state=42)

#     # Define the objective function for optimization
#     def objective(params):
#         num_atoms, sparsity_level = params
#         sparse_codes, dictionary, reconstruction_error = gpu_ksvd(flattened_images, num_atoms, sparsity_level)
#         return float(reconstruction_error)  # Minimize reconstruction error

#     best_score = float('inf')
#     best_params = None

#     with tqdm(total=n_calls, desc="Optimizing K-SVD") as pbar:
#         for _ in range(n_calls):
#             suggested_params = optimizer.ask()
#             score = objective(suggested_params)
#             optimizer.tell(suggested_params, score)

#             if score < best_score:
#                 best_score = score
#                 best_params = suggested_params

#             pbar.update(1)

#     best_num_atoms, best_sparsity_level = best_params
#     sparse_codes, best_dictionary, _ = gpu_ksvd(flattened_images, best_num_atoms, best_sparsity_level)

#     final_params = {
#         'num_atoms': best_num_atoms,
#         'sparsity_level': best_sparsity_level
#     }
#     print(f"Best K-SVD Params: {final_params}")

#     return sparse_codes, best_dictionary, final_params


# class KSVDDictionary:
#     def __init__(self, dictionary, image_shape):
#         self.dictionary = cp.array(dictionary).T  # Store dictionary on the GPU
#         self.image_shape = image_shape

#     def reconstruct(self, sparse_codes):
#         if isinstance(sparse_codes, torch.Tensor):
#             sparse_codes = cp.array(sparse_codes.detach().cpu().numpy())  # Convert to cupy array if needed
#         elif isinstance(sparse_codes, np.ndarray):
#             sparse_codes = cp.array(sparse_codes)

#         # Check for alignment
#         if sparse_codes.shape[1] != self.dictionary.shape[0]:
#             raise ValueError(f"Shape mismatch: sparse_codes {sparse_codes.shape} and dictionary {self.dictionary.shape}")

#         # Perform reconstruction on the GPU
#         reconstructed_flat = cp.dot(sparse_codes, self.dictionary)
#         reconstructed_images = reconstructed_flat.reshape((-1, *self.image_shape))
#         # print("Sparse Codes:", sparse_codes)  # Check if sparse codes contain meaningful values
#         # print("Reconstructed Images (flat):", reconstructed_flat)  # Check reconstructed values before reshaping
#         # Convert back to torch tensor for compatibility with PyTorch (move to CPU if needed)
#         return torch.tensor(reconstructed_images.get(), dtype=torch.float32)

'''
Borderline
'''

# def optimize_ksvd(images, input_size, n_calls=20):
#     # Unpack input size
#     num_channels, height, width = input_size
#     num_samples = images.shape[0]
    
#     # Reshape images to have (N, C, H*W) shape for channel-wise processing
#     flattened_images = images.reshape(num_samples, num_channels, -1)  # Shape: (N, C, H*W)

#     # Define search space for KSVD parameters
#     search_space = [
#         Integer(350, 400, name='num_atoms'),       # Range for num_atoms
#         Integer(22, 40, name='sparsity_level')     # Range for sparsity_level
#     ]
    
#     # Initialize optimizer
#     optimizer = Optimizer(search_space, random_state=42)
    
#     # Objective function for each parameter set
#     def objective(params):
#         num_atoms, sparsity_level = params
#         reconstruction_errors = []

#         for c in range(num_channels):
#             channel_data = flattened_images[:, c]

#             # Debugging: Check the shape and basic statistics of the channel data
#             print(f"Processing Channel {c}")
#             print(f"Shape: {channel_data.shape}")
#             print(f"Mean: {channel_data.mean():.5f}, Variance: {channel_data.var():.5f}")
            
#             # Optionally add a small amount of noise if variance is very low (commented by default)
#             # if channel_data.var() < 1e-8:
#             #     channel_data += np.random.normal(0, 1e-4, size=channel_data.shape)
#             #     print("Added small noise to channel data to avoid zero variance.")

#             dict_learner = DictionaryLearning(
#                 n_components=num_atoms,
#                 transform_algorithm='omp',
#                 transform_n_nonzero_coefs=sparsity_level,
#                 max_iter=30
#             )
#             try:
#                 sparse_codes = dict_learner.fit_transform(channel_data)
#                 reconstructed_channel = np.dot(sparse_codes, dict_learner.components_)
#                 reconstruction_error = np.mean((channel_data - reconstructed_channel) ** 2)
#                 reconstruction_errors.append(reconstruction_error)
#             except ValueError as e:
#                 print(f"Error in sparse encoding for channel {c}: {e}")
#                 return float('inf')  # Return a high penalty score if an error occurs

#         # Average reconstruction error across channels
#         return np.mean(reconstruction_errors)

#     # Run Bayesian optimization with tqdm progress bar
#     best_score = float('inf')
#     best_params = None
#     with tqdm(total=n_calls, desc="Optimizing K-SVD") as pbar:
#         for _ in range(n_calls):
#             suggested_params = optimizer.ask()
#             score = objective(suggested_params)
#             optimizer.tell(suggested_params, score)
            
#             if score < best_score:
#                 best_score = score
#                 best_params = suggested_params
            
#             pbar.update(1)
    
#     best_num_atoms, best_sparsity_level = best_params
    
#     # Get the final best sparse codes and dictionary per channel
#     best_sparse_codes, best_dictionaries = [], []
#     for c in range(num_channels):
#         dict_learner = DictionaryLearning(
#             n_components=best_num_atoms,
#             transform_algorithm='omp',
#             transform_n_nonzero_coefs=best_sparsity_level,
#             max_iter=20
#         )
#         sparse_codes = dict_learner.fit_transform(flattened_images[:, c])
#         best_sparse_codes.append(sparse_codes)
#         best_dictionaries.append(dict_learner.components_)

#     final_params = {
#         'num_atoms': best_num_atoms,
#         'sparsity_level': best_sparsity_level
#     }
#     print(f"Best K-SVD Params: {final_params}")
    
#     return best_sparse_codes, best_dictionaries, final_params

# class KSVDDictionary:
#     def __init__(self, dictionaries, image_shape):
#         self.dictionaries = dictionaries  # List of dictionaries, one for each channel
#         self.image_shape = image_shape    # Expected shape of (C, H, W)

#     def reconstruct(self, sparse_codes):
#         if isinstance(sparse_codes, torch.Tensor):
#             sparse_codes = sparse_codes.detach().cpu().numpy()

#         num_channels = len(self.dictionaries)  # Should be 3 for RGB
#         num_images = len(sparse_codes[0])  # Assuming sparse_codes[c] is per channel and contains all images for that channel
#         reconstructed_images = []

#         # Loop through each image
#         for i in range(num_images):
#             image_reconstructed_channels = []
            
#             # Loop through each channel to reconstruct the image
#             for c in range(num_channels):
#                 dictionary = self.dictionaries[c]
#                 channel_sparse_codes = sparse_codes[c][i]  # Sparse code for this image's channel
                
#                 # Ensure shape alignment between sparse code and dictionary
#                 if channel_sparse_codes.shape[0] != dictionary.shape[0]:
#                     raise ValueError(f"Shape mismatch for channel {c}: sparse_codes {channel_sparse_codes.shape} and dictionary {dictionary.shape}")

#                 # Reconstruct this channel for the current image
#                 reconstructed_flat = np.dot(channel_sparse_codes, dictionary)
#                 reconstructed_channel = reconstructed_flat.reshape(*self.image_shape[1:])  # Reshape to (H, W)
#                 image_reconstructed_channels.append(reconstructed_channel)

#             # Stack channels for this image to create (3, H, W)
#             reconstructed_image = np.stack(image_reconstructed_channels, axis=0)
#             reconstructed_images.append(reconstructed_image)

#         # Convert to tensor with shape (N, C, H, W)
#         print("Length", len(reconstructed_images))
#         return torch.tensor(np.stack(reconstructed_images), dtype=torch.float32)



