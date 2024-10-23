# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import time

from replay_buffer import ReplayBuffer
from ksvd_compression import optimize_ksvd, KSVDDictionary
from kmeans_sampling import KMeansOptimizer
from image_customization import preprocess_images
from autoencoder_compression import NeuralImageCompression, Encoder, Decoder
from pca_compression import PCACompression
from csc_compression import ConvolutionalSparseCoding
from dpp_sampling import DPP

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dynamically calculate the size of the flattened features after conv and pooling layers
        self.to_linear = None
        self._get_conv_output(input_size)
        
        self.fc1 = nn.Linear(self.to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.to_linear)  # Flatten the output of conv layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _get_conv_output(self, input_size):
        # Dummy pass to get the output size
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *input_size))
        output_feat = self.features(input)
        self.to_linear = int(torch.numel(output_feat) / bs)

    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

def train_model(model, train_loader, replay_buffer, criterion, optimizer, device, task_id, batch_size, compression_name=None):
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0

    # Load all task data at once
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    # Retrieve replay data for previous tasks if not Task 0
    if task_id != 0:
        replay_data, replay_target = replay_buffer.get_replay_data(task_id)
        replay_data, replay_target = replay_data.to(device), replay_target.to(device)
        ### KSVD has different input size
        if compression_name == "csc":
            replay_data = replay_data.squeeze(1)
        elif compression_name == "pca":
            replay_data = replay_data.unsqueeze(1)
        else:
            pass
        
        # Concatenate task data and replay data
        print(f"Data Shape: {data.size()}, Replay Data Shape: {replay_data.size()}")
        combined_data = torch.cat([data, replay_data], dim=0)
        combined_target = torch.cat([target, replay_target], dim=0)

        # Shuffle combined dataset
        indices = torch.randperm(combined_data.size(0))
        combined_data, combined_target = combined_data[indices], combined_target[indices]
    else:
        combined_data, combined_target = data, target

    # Create mini-batches manually
    print(f"Combined Data Shape: {combined_data.size()}")
    for i in range(0, combined_data.size(0), batch_size):
        combined_data = combined_data.detach()
        combined_target = combined_target.detach()
        mini_batch_data = combined_data[i:i + batch_size]
        mini_batch_target = combined_target[i:i + batch_size]

        # Training step
        outputs = model(mini_batch_data)
        loss = criterion(outputs, mini_batch_target)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Track accuracy and loss
        _, predicted = torch.max(outputs.data, 1)
        total_samples += mini_batch_target.size(0)
        total_correct += (predicted == mini_batch_target).sum().item()
        total_loss += loss.item() * mini_batch_data.size(0)
    
    torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples
    training_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
    print(f"Training Accuracy: {training_accuracy:.2f}%")
    return model, avg_loss



def main(params):
    # Unpack parameters
    dataset_name = params['dataset_name']
    num_tasks = params['num_tasks']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    buffer_capacity_per_task = params['buffer_capacity_per_task']
    input_size = params['input_size']
    num_classes_per_task = params['num_classes_per_task']
    input_channels = params['input_channels']
    compression = params['compression']
    sampling = params['sampling']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validation_accuracies = {}
    total_classes = num_classes_per_task * num_tasks
    model = CNN(input_channels, total_classes, input_size).to(device)
    
    # Define loss and optimizer
    train_criterion = nn.CrossEntropyLoss(reduction='mean')
    eval_criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity_per_task=buffer_capacity_per_task, input_shape=input_size)
    time_sampling = []
    time_compression = []
    
    torch.cuda.empty_cache()
    for task_id in range(num_tasks):
        print(f'Training on Task {task_id+1}/{num_tasks}')
        train_dataset, input_shape = preprocess_images(dataset_name=dataset_name, task_id=task_id, train=True)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        
        for epoch in range(num_epochs):
            model, avg_loss = train_model(model, train_loader, replay_buffer, train_criterion, optimizer, device, task_id, batch_size, compression_name=compression)
            print(f'Epoch {epoch+1}/{num_epochs}, Avg. Loss: {avg_loss:.4f}')

        loss_values = compute_loss_values(model, train_loader, eval_criterion, device)
        all_features = extract_features(model, train_loader, device, input_size)
        
        start_time_sampling = time.time()
        if sampling == 'kmeans':
            data, target = next(iter(train_loader))
            data, target = data.to(device), target.to(device)
            kmeans_optimizer = KMeansOptimizer(features=all_features, targets=target, loss_values=loss_values, n_calls=20, max_retrieval=buffer_capacity_per_task, input_shape=input_size)
            selected_indices, best_params = kmeans_optimizer.run_optimization()
        
        elif sampling == 'dpp':
            optimal_sigma = DPP.hyperparameter_tuning(all_features, n_trials=20, device=device)
            kernel_matrix = DPP.compute_rbf_kernel(all_features, sigma=optimal_sigma, device=device)
            dpp = DPP(kernel_matrix, device=device, max_samples=buffer_capacity_per_task)
            selected_indices = dpp.sample()
            
        else:
            selected_indices = np.random.choice(len(train_dataset), buffer_capacity_per_task, replace=False)
        
        end_time_sampling = time.time()
        sampling_time = end_time_sampling - start_time_sampling
        time_sampling.append(sampling_time)
        selected_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
        data_loader = DataLoader(selected_dataset, batch_size=len(selected_dataset))
        images, labels = next(iter(data_loader))


        start_time_compression = time.time()
        if compression == "jpeg":
            compressed_data = [(tensor_to_jpeg(img), label) for img, label in zip(images, labels)]
            replay_buffer.add(task_id=task_id, compressed_data=compressed_data, compression_name='jpeg')
            visualize_jpeg_compressed_images(images, compressed_data, './visualizations')
            
        elif compression == 'ksvd':
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            flattened_images = images.reshape(images.shape[0], -1)
            best_sparse_codes, best_dictionary, best_ksvd_params = optimize_ksvd(flattened_images, n_calls=20)
            # compressed_data = [([torch.tensor(best_sparse_codes[c][i]) for c in range(3)], label)
            #            for i, label in enumerate(labels)] 
            compressed_data = [(torch.tensor(sparse_code), label) for sparse_code, label in zip(best_sparse_codes, labels)]  
            replay_buffer.add(task_id=task_id, compressed_data=compressed_data, dictionary=best_dictionary, compression_name='ksvd')
            visualize_compressed_images(images, compressed_data, best_dictionary, './visualizations')

        elif compression == 'pca':
            images = images.to(device)
            pca_components, whiten, svd_solver = PCACompression.tune_hyperparameters(images, input_shape=input_size, device=device)
            pca_compression = PCACompression(n_components=pca_components, whiten=whiten, svd_solver=svd_solver, device=device)
            pca_compression.fit(images)
            pca_compression.save_pca_params(task_id)
            compressed_data = [(pca_compression.compress(img.unsqueeze(0)), label) for img, label in zip(images, labels)]
            replay_buffer.add(task_id=task_id, compressed_data=compressed_data, hyperparameter={'n_components': pca_components, 'whiten': whiten, 'svd_solver': svd_solver}, compression_name='pca')
            visualize_pca_compressed_images(images, compressed_data, pca_compression, './visualizations', input_size)

        elif compression == 'csc':
            images = images.to(device)
            best_params = ConvolutionalSparseCoding.tune_hyperparameters(data_loader, input_shape=input_size, device=device)
            num_filters = best_params["num_filters"]
            kernel_size = best_params["kernel_size"]
            sparsity_weight = best_params["sparsity_weight"]
            csc = ConvolutionalSparseCoding(num_filters=num_filters, kernel_size=kernel_size, sparsity_weight=sparsity_weight, device=device)
            csc.train(train_loader=data_loader, num_epochs=10, learning_rate=1e-3)
            compressed_data = [(csc.compress(img.unsqueeze(0)), label) for img, label in zip(images, labels)]
            replay_buffer.add(task_id=task_id, compressed_data=compressed_data, dictionary=csc.conv_dict.weight.clone().detach(), hyperparameter=best_params, compression_name="csc")
            visualize_csc_compressed_images(original_images=images, compressed_data=compressed_data, input_shape=input_size, csc=csc, device=device, save_path='./visualizations')
        
        end_time_compression = time.time()
        compression_time = end_time_compression - start_time_compression
        time_compression.append(compression_time)
        
        total_memory = replay_buffer.calculate_memory_usage()
        print(f"Total Memory Usage of Replay Buffer: {total_memory} B")
        
        validation_accuracies[task_id] = []
        for validate_task in range(task_id + 1):
            validation_dataset, input_shape = preprocess_images(dataset_name=dataset_name, task_id=validate_task, train=False)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
            validation_accuracy = evaluate_model(model, validation_loader, device)
            validation_accuracies[task_id].append(validation_accuracy)
            print(f'Validation Accuracy for Task {validate_task + 1}: {validation_accuracy:.2f}%')
        
        print("All Validation Accuracies:", validation_accuracies)
        print("All times for sampling:", time_sampling)
        print("All times for compression:", time_compression)
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

def compute_loss_values(model, data_loader, criterion, device):
    model.eval()
    loss_values = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            losses = criterion(outputs, target)  # This should be a tensor of losses, one per batch item

            # Ensure criterion is set to reduction='none' to make losses a tensor
            losses = losses.detach().cpu().numpy()  # Convert tensor of losses to a numpy array
            loss_values.extend(losses)  # Extend the list by appending all individual losses

    return loss_values


def extract_features(model, data_loader, device, input_shape):
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            
            # Forward pass up to the final convolutional layer
            conv_output = model.features(data)
            flattened_output = conv_output.view(conv_output.size(0), -1)  # Flatten if needed
            
            # Pass through the fully connected layers, if applicable, or use flattened conv output directly
            features = model.fc1(flattened_output) if hasattr(model, 'fc1') else flattened_output
            features_list.append(features.cpu().numpy())
    
    all_features = np.concatenate(features_list, axis=0)
    print(f"Extracted Features Shape: {all_features.shape}")
    return all_features

# def extract_features(model, data_loader, device, input_shape):
#     model.eval()
#     features_list = []

#     with torch.no_grad():
#         for data, _ in data_loader:
#             data = data.to(device)
            
#             # Forward pass up to the final convolutional layer
#             conv_output = model.features(data)
#             flattened_output = conv_output.view(conv_output.size(0), -1)  # Flatten if needed
            
#             # Pass through the fully connected layers, if applicable, or use flattened conv output directly
#             features = model.fc1(flattened_output) if hasattr(model, 'fc1') else flattened_output
#             features_list.append(features.cpu().numpy())

#     all_features = np.concatenate(features_list, axis=0)
#     print(f"Extracted Features Shape: {all_features.shape}")
#     return all_features



def evaluate_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()
    accuracy = 100 * total_correct / total_samples
    return accuracy

def visualize_jpeg_compressed_images(original_images, compressed_data, save_path, num_images=2):
    fig, axes = plt.subplots(2, num_images, figsize=(10, 5))
    for i in range(num_images):
        axes[0, i].imshow(original_images[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        decompressed_image = jpeg_to_tensor(compressed_data[i][0])
        axes[1, i].imshow(decompressed_image.squeeze().numpy(), cmap='gray')
        axes[1, i].set_title("Decompressed")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path}/jpeg_comparison_top_{num_images}.png")
    plt.close()

def visualize_compressed_images(original_images, compressed_data, dictionary, save_path, num_images=2):
    ksvd = KSVDDictionary(dictionary=dictionary, image_shape=(28, 28))  # Adjust image_shape as needed
    # Ensure you extract only the sparse codes from the tuples
    sparse_codes = [data for data, _ in compressed_data[:num_images]]
    reconstructed_images = [
        ksvd.reconstruct(data.clone().detach().unsqueeze(0)).squeeze(0).numpy()
        for data in sparse_codes
    ]


    fig, axes = plt.subplots(2, num_images, figsize=(10, 5))
    for i in range(num_images):
        axes[0, i].imshow(original_images[i].reshape(28, 28), cmap='gray')  # Adjust reshape as per image dimensions
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path}/comparison_top_{num_images}.png")
    plt.close()


# def visualize_compressed_images(original_images, compressed_data, dictionary, save_path, num_images=2):
#     channels = original_images.shape[1] if len(original_images.shape) > 3 else 1
#     image_shape = (channels, *original_images.shape[-2:])
#     ksvd = KSVDDictionary(dictionaries=dictionary, image_shape=image_shape)

#     # Extract and structure sparse_codes per image as [[[R], [G], [B]], ...]
#     sparse_codes = [torch.stack([data[c] for data, _ in compressed_data[:num_images]]) for c in range(3)]

#     # Reconstruct images from sparse codes
#     reconstructed_images = ksvd.reconstruct(sparse_codes).numpy()

#     fig, axes = plt.subplots(2, num_images, figsize=(10, 5))
#     for i in range(num_images):
#         original_image = original_images[i].transpose(1, 2, 0) if channels == 3 else original_images[i].squeeze()
#         axes[0, i].imshow(original_image, cmap='gray' if channels == 1 else None)
#         axes[0, i].set_title("Original")
#         axes[0, i].axis('off')

#         reconstructed_image = reconstructed_images[i].transpose(1, 2, 0) if channels == 3 else reconstructed_images[i].squeeze()
#         axes[1, i].imshow(reconstructed_image, cmap='gray' if channels == 1 else None)
#         axes[1, i].set_title("Reconstructed")
#         axes[1, i].axis('off')

#     plt.tight_layout()
#     plt.savefig(f"{save_path}/comparison_top_{num_images}.png")
#     plt.close()


'''

'''
def tensor_to_jpeg(tensor, quality=80):
    """Convert a PyTorch tensor to a JPEG image (in bytes)."""
    # Normalize the tensor to [0, 255] and convert to PIL image
    tensor = tensor.squeeze().cpu()  # Remove batch dim, if any
    img = transforms.ToPILImage()(tensor)
    
    # Save as JPEG in-memory using BytesIO
    with io.BytesIO() as output:
        img.save(output, format="JPEG", quality=quality)
        jpeg_bytes = output.getvalue()
    return jpeg_bytes

def jpeg_to_tensor(jpeg_bytes):
    """Convert a JPEG image (in bytes) back to a PyTorch tensor."""
    with io.BytesIO(jpeg_bytes) as input_bytes:
        img = Image.open(input_bytes)
        tensor = transforms.ToTensor()(img).unsqueeze(0)  # Add batch dim
    return tensor


def visualize_autoencoder_images(original_images, decompressed_images, save_path, num_images=5):
    fig, axes = plt.subplots(2, num_images, figsize=(10, 4))
    for i in range(num_images):
        axes[0, i].imshow(original_images[i].view(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(decompressed_images[i][0].view(28, 28), cmap="gray")
        axes[1, i].axis("off")
    plt.savefig(save_path)
    plt.show()

def visualize_pca_compressed_images(original_images, compressed_data, pca_compression, save_path, input_shape, num_images=2):
    reconstructed_images = [
        pca_compression.decompress(data.clone().detach(), shape=input_shape).squeeze().cpu().numpy()
        for data, _ in compressed_data[:num_images]
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
    plt.savefig(f"{save_path}/pca_comparison_top_{num_images}.png")
    plt.close()

def visualize_csc_compressed_images(original_images, compressed_data, input_shape, csc, device, save_path, num_images=2):
    reconstructed_images = [csc.decompress([data[0]], input_shape).squeeze().detach().cpu().numpy() for data, _ in compressed_data[:num_images]]

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


if __name__ == '__main__':
    # Define all parameters in a dictionary
    params = {
        'dataset_name': 'MNIST',
        'num_tasks': 5,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'batch_size': 32,
        'buffer_capacity_per_task': 750,  # Capacity per task
        'input_size': (1, 28, 28),  # Adjust based on dataset
        'num_classes_per_task': 2,
        'input_channels': 1, 
        'compression': 'jpeg',  # Choose from 'ksvd', 'jpeg', 'pca', 'csc'
        'sampling': 'dpp', # Choose from 'kmeans', 'dpp', None
    }
    main(params)
