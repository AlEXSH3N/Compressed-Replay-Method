import torch
from torchvision import datasets, transforms

def preprocess_images(dataset_name, task_id, train=True):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any additional preprocessing steps here
        ])
        # Define class splits per task
        task_classes = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7],
            4: [8, 9]
        }
        classes = task_classes[task_id]
        
        full_dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        # Filter dataset for the current task classes
        indices = [i for i, (_, label) in enumerate(full_dataset) if label in classes]
        task_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Preprocessed Dataset Size: {len(task_dataset)}, Task ID: {task_id}")
        input_shape = (1, 28, 28)
        
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any additional preprocessing steps here
        ])
        # Define class splits per task for CIFAR100
        task_classes = {
            0: [0, 1],  # Task 0: Classes 0 and 1
            1: [2, 3],  # Task 1: Classes 2 and 3
            2: [4, 5],  # Task 2: Classes 4 and 5
            3: [6, 7],  # Task 3: Classes 6 and 7
            4: [8, 9]   # Task 4: Classes 8 and 9
        }
        classes = task_classes[task_id]

        full_dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        # Filter dataset for the current task classes
        indices = [i for i, (_, label) in enumerate(full_dataset) if label in classes]
        task_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Preprocessed Dataset Size: {len(task_dataset)}, Task ID: {task_id}")
        input_shape = (3, 32, 32)
    else:
        raise ValueError('Dataset not supported')
    
    return task_dataset, input_shape
