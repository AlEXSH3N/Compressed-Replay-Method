import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
import io
from PIL import Image
from torchvision import transforms

from ksvd_compression import KSVDDictionary
from autoencoder_compression import NeuralImageCompression
from pca_compression import PCACompression
from csc_compression import ConvolutionalSparseCoding

class ReplayBuffer:
    def __init__(self, capacity_per_task, input_shape = None):
        self.capacity_per_task = capacity_per_task
        self.buffer = {}
        self.dictionaries = {}  # Store task dictionaries for reference
        self.input_shape = input_shape  # Store the shape of the input data for reconstruction
        self.compression_type = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters = {}
        self.convolution_weights = {}

    
    def add(self, task_id, compressed_data, dictionary=None, hyperparameter=None, compression_name: str = "ksvd"):
        self.buffer[task_id] = []
        self.compression_type = compression_name
        if compression_name == "ksvd":
            self.dictionaries[task_id] = dictionary
        elif compression_name == "csc":
            self.convolution_weights[task_id] = dictionary
        if hyperparameter is not None:
            self.hyperparameters[task_id] = hyperparameter
        self.buffer[task_id].extend(compressed_data)
        if len(self.buffer[task_id]) > self.capacity_per_task:
            self.buffer[task_id] = self.buffer[task_id][-self.capacity_per_task:]
    
    def get_replay_data(self, current_task_id):
        # Collect all samples from tasks up to the current_task_id
        replay_data = []
        targets = []
        counter = 0

        for task_id in range(current_task_id + 1):
            if task_id in self.buffer:
                task_reconstructed_data = []
                task_targets = []
                
                if self.compression_type == "ksvd":
                    dictionary = self.dictionaries[task_id]
                    ksvd = KSVDDictionary(dictionary, self.input_shape)
                    task_compressed_data, task_targets = zip(*self.buffer[task_id])
                    task_compressed_data = torch.stack(task_compressed_data)  # Stack compressed data for batch processing
                    task_reconstructed_data = ksvd.reconstruct(task_compressed_data)
                
                elif self.compression_type == "pca":
                    n_components = self.hyperparameters[task_id]["n_components"]
                    whiten = self.hyperparameters[task_id]["whiten"]
                    svd_solver = self.hyperparameters[task_id]["svd_solver"]
                    pca_compression = PCACompression(n_components=n_components, whiten=whiten, svd_solver=svd_solver, device=self.device)
                    pca_compression.pca = PCACompression.load_pca_params(f"pca_compressed_metadata/pca_params_task_{task_id}.pkl", n_components=n_components)
                    task_compressed_data, task_targets = zip(*self.buffer[task_id])
                    task_compressed_data = torch.stack(task_compressed_data).to(self.device)
                    task_reconstructed_data = [
                        pca_compression.decompress(data.unsqueeze(0), shape=self.input_shape).squeeze()
                        for data in task_compressed_data
                    ]
                
                elif self.compression_type == "csc":
                    num_filters = self.hyperparameters[task_id]["num_filters"]
                    kernel_size = self.hyperparameters[task_id]["kernel_size"]
                    sparsity_weight = self.hyperparameters[task_id]["sparsity_weight"]
                    csc = ConvolutionalSparseCoding(num_filters=num_filters, kernel_size=kernel_size, sparsity_weight=sparsity_weight, device=self.device)
                    csc.conv_dict.weight = torch.nn.Parameter(self.convolution_weights[task_id])
                    task_compressed_data, task_targets = zip(*self.buffer[task_id])
                    for sparse_code in task_compressed_data:
                        reconstructed = csc.decompress(sparse_code, input_shape=self.input_shape)
                        task_reconstructed_data.append(reconstructed)
                    
                elif self.compression_type == "jpeg":
                    for jpeg_images, label in self.buffer[task_id]:
                        tensor_images = self.jpeg_to_tensor(jpeg_images)
                        task_reconstructed_data.extend(tensor_images)
                        task_targets.append(label)

                # Append reconstructed images and corresponding targets
                # if error for ksvd, convert replay data to append
                replay_data.extend(task_reconstructed_data)
                targets.extend(task_targets)

        if replay_data:
            # if error for ksvd, convert replay data torch to cat
            replay_data = torch.stack(replay_data)  # Concatenate all reconstructed data
            targets = torch.tensor(targets)
            return replay_data, targets
        
        return None, None
    
    def get_replay_latent_data(self, current_task_id, device):
        replay_latent = []
        replay_targets = []
        for task_id in range(current_task_id):
            if task_id in self.buffer:
                task_data = [data.to(device) for data, _ in self.buffer[task_id]]  # Move data to the correct device
                task_targets = [target.to(device) for _, target in self.buffer[task_id]]  # Move targets to the correct device
                replay_latent.extend(task_data)
                replay_targets.extend(task_targets)

        return torch.stack(replay_latent), torch.stack(replay_targets)


    def is_empty(self):
        return not any(self.buffer.values())
    
    
    def calculate_memory_usage(self):
        total_memory = 0

        # Measure memory usage of buffer
        buffer_memory = sum(self._get_memory_size(data) for task_data in self.buffer.values() for data, _ in task_data)
        total_memory += buffer_memory

        # Measure memory usage of dictionaries
        dictionaries_memory = sum(self._get_memory_size(dictionary) for dictionary in self.dictionaries.values())
        total_memory += dictionaries_memory

        # Measure memory usage of convolutional weights (for CSC)
        conv_weights_memory = sum(self._get_memory_size(weight) for weight in self.convolution_weights.values())
        total_memory += conv_weights_memory

        print(f"Buffer Memory Usage: {buffer_memory} B")
        print(f"Dictionaries Memory Usage: {dictionaries_memory} B")
        print(f"Convolutional Weights Memory Usage: {conv_weights_memory} B")
        print(f"Total Replay Buffer Memory Usage: {total_memory} B")

        return total_memory
    
    def _get_memory_size(self, obj):
        """Calculate memory size of an object."""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            return sys.getsizeof(obj)
    
    def jpeg_to_tensor(self, jpeg_bytes):
        """Convert a JPEG image (in bytes) back to a PyTorch tensor."""
        with io.BytesIO(jpeg_bytes) as input_bytes:
            img = Image.open(input_bytes)
            tensor = transforms.ToTensor()(img).unsqueeze(0)  # Add batch dim
        return tensor
    
