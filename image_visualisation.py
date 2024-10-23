import torch
from torch.utils.data import DataLoader, random_split
import random
import matplotlib.pyplot as plt

import scipy.sparse as sp
from replay_buffer import ReplayBuffer
from ksvd_compression import optimize_ksvd, KSVDDictionary
from kmeans_sampling import KMeansOptimizer
from image_customization import preprocess_images
from autoencoder_compression import Autoencoder
from pca_compression import PCACompression
from csc_compression import ConvolutionalSparseCoding

# Main function for compression and visualization
def main(params):
    dataset_name = params['dataset_name']
    buffer_capacity = 750
    compression_method = params['compression']
    input_size = params['input_size']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset, input_shape = preprocess_images(dataset_name=dataset_name, task_id=0, train=True)
    subset_indices = random.sample(range(len(dataset)), buffer_capacity)  # Randomly select 750 images
    subset = torch.utils.data.Subset(dataset, subset_indices)
    data_loader = DataLoader(subset, batch_size=len(subset))
    images, labels = next(iter(data_loader))
    
    # Run compression and visualization for specified method
    if compression_method == 'ksvd':
        run_ksvd(images, labels, input_shape, './visualizations', num_images=2)
    elif compression_method == 'pca':
        run_pca(images, labels, input_shape, './visualizations', num_images=2, device=device)
    elif compression_method == 'csc':
        run_csc(images, labels, input_shape, './visualizations', num_images=2, device=device)
    else:
        print("Invalid compression method")

# Function for KSVD compression and visualization
def run_ksvd(images, labels, input_shape, save_path, num_images=2):
    flattened_images = images.reshape(images.shape[0], -1)
    best_sparse_codes, best_dictionary, best_ksvd_params = optimize_ksvd(flattened_images, n_calls=25)
    compressed_data = [(torch.tensor(sparse_code), label) for sparse_code, label in zip(best_sparse_codes, labels)]
    visualize_compressed_images(images, compressed_data, best_dictionary, save_path, num_images=num_images)

# Function for PCA compression and visualization
def run_pca(images, labels, input_shape, save_path, num_images=2, device='cpu'):
    images = images.to(device)
    pca_components, whiten, svd_solver = PCACompression.tune_hyperparameters(images, input_shape=input_shape, device=device)
    pca_compression = PCACompression(n_components=pca_components, whiten=whiten, svd_solver=svd_solver, device=device)
    pca_compression.fit(images)
    compressed_data = [(pca_compression.compress(img.unsqueeze(0)), label) for img, label in zip(images, labels)]
    visualize_pca_compressed_images(images, compressed_data, pca_compression, save_path, input_shape, num_images=num_images)

# Function for CSC compression and visualization
def run_csc(images, labels, input_shape, save_path, num_images=2, device='cpu'):
    images = images.to(device)
    best_params = ConvolutionalSparseCoding.tune_hyperparameters(DataLoader(images), input_shape=input_shape, device=device)
    csc = ConvolutionalSparseCoding(num_filters=best_params["num_filters"], kernel_size=best_params["kernel_size"], sparsity_weight=best_params["sparsity_weight"], device=device)
    csc.train(DataLoader(images), num_epochs=10, learning_rate=1e-3)
    compressed_data = [csc.compress(img.unsqueeze(0)) for img, label in zip(images, labels)]
    visualize_csc_compressed_images(images, compressed_data, input_shape, csc, device, save_path, num_images=num_images)

# Visualization function for KSVD
def visualize_compressed_images(original_images, compressed_data, dictionary, save_path, num_images=2):
    ksvd = KSVDDictionary(dictionary=dictionary, image_shape=(28, 28))
    sparse_codes = [data for data, _ in compressed_data[:num_images]]
    reconstructed_images = [ksvd.reconstruct(data.clone().detach().unsqueeze(0)).squeeze(0).numpy() for data in sparse_codes]

    fig, axes = plt.subplots(2, num_images, figsize=(10, 5))
    for i in range(num_images):
        axes[0, i].imshow(original_images[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_path}/ksvd_comparison_top_{num_images}.png")
    plt.close()

# Visualization function for PCA
def visualize_pca_compressed_images(original_images, compressed_data, pca_compression, save_path, input_shape, num_images=2):
    # Decompress the compressed images using PCA
    reconstructed_images = [
        pca_compression.decompress(data.clone().detach(), shape=input_shape).squeeze().cpu().numpy()
        for data, _ in compressed_data[:num_images]
    ]

    fig, axes = plt.subplots(2, num_images, figsize=(10, 5))
    for i in range(num_images):
        # Display the original image
        original_image = original_images[i].cpu().numpy().transpose(1, 2, 0)  # Reshape for RGB if CIFAR-10
        axes[0, i].imshow(original_image)
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Display the reconstructed image
        reconstructed_image = reconstructed_images[i].transpose(1, 2, 0)  # Shape should now be (H, W, C) for RGB
        axes[1, i].imshow(reconstructed_image)
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path}/csc_comparison_top_{num_images}_last.png")
    plt.close()

# Visualization function for CSC
def visualize_csc_compressed_images(original_images, compressed_data, input_shape, csc, device, save_path, num_images=2):
    reconstructed_images = [
        csc.decompress([torch.tensor(data.toarray(), device=device)], input_shape).squeeze().cpu().numpy()
        if isinstance(data, sp.csr_matrix) else csc.decompress([data], input_shape).squeeze().cpu().detach().numpy()
        for data in compressed_data[:num_images]
    ]
    
    fig, axes = plt.subplots(2, num_images, figsize=(10, 5))
    for i in range(num_images):
        axes[0, i].imshow(original_images[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_path}/csc_comparison_top_{num_images}.png")
    plt.close()

# Parameters to run the main function
params = {
    'dataset_name': 'CIFAR10',  # or CIFAR-10
    'compression': 'pca',    # 'pca' or 'csc' depending on which to test
    'input_size': (3, 32, 32)    # Adjust based on the dataset and image size
}
main(params)
