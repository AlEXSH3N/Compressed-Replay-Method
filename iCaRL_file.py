import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict, deque
from image_customization import preprocess_images
from sklearn.utils import shuffle


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
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
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *input_size))
        output_feat = self.features(input)
        self.to_linear = int(torch.numel(output_feat) / bs)

    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class ReplayBuffer:
    def __init__(self, capacity=750):
        self.buffer = defaultdict(deque)
        self.capacity = capacity

    def add_exemplars(self, task_id, exemplars):
        # Add exemplars and trim to capacity if exceeded
        self.buffer[task_id].extend([
            (torch.tensor(data, dtype=torch.float32).view(1, 28, 28), target)
            for data, target in exemplars
        ])
        if len(self.buffer[task_id]) > self.capacity:
            self.buffer[task_id] = deque(list(self.buffer[task_id])[-self.capacity:], maxlen=self.capacity)

    def get_replay_data(self, batch_size):
        replay_data, replay_target = [], []
        # Gather data from all tasks
        for task_id in self.buffer:
            task_data = [data for data, _ in self.buffer[task_id]]
            task_target = [target for _, target in self.buffer[task_id]]
            replay_data.extend(task_data)
            replay_target.extend(task_target)

        # Convert to tensors and limit to batch size
        replay_data = torch.cat(replay_data[:batch_size], dim=0)
        replay_target = torch.tensor(replay_target[:batch_size], dtype=torch.long)
        return replay_data, replay_target

### Updated KMeansExemplarSelector

class KMeansExemplarSelector:
    def __init__(self, features, targets, loss_values, max_retrieval=750, random_state=42):
        self.features = features.view(features.size(0), -1).cpu().numpy() if isinstance(features, torch.Tensor) else features.reshape(features.shape[0], -1)
        self.targets = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        self.loss_values = np.array(loss_values)
        self.max_retrieval = max_retrieval
        self.random_state = random_state

    def select_exemplars(self, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(self.features)
        return self._select_stratified_samples(cluster_labels, num_clusters)

    def _select_stratified_samples(self, cluster_labels, instances_per_cluster):
        unique_labels = np.unique(self.targets)
        selected_indices = set()

        for label in unique_labels:
            indices = np.where(self.targets == label)[0]
            label_cluster_labels = cluster_labels[indices]
            label_loss_values = self.loss_values[indices]

            for cluster in np.unique(label_cluster_labels):
                cluster_indices = np.where(label_cluster_labels == cluster)[0]
                cluster_losses = label_loss_values[cluster_indices]
                sorted_loss_indices = cluster_indices[np.argsort(cluster_losses)]

                # Select exemplars from each cluster, evenly distributing across classes
                selected_indices.update(indices[sorted_loss_indices[:instances_per_cluster]].tolist())

        selected_indices = np.array(list(selected_indices))
        selected_indices = shuffle(selected_indices, random_state=self.random_state)
        selected_indices = selected_indices[:min(len(selected_indices), self.max_retrieval)]

        print(f"Selected {len(selected_indices)} samples after balancing across tasks.")
        return selected_indices

# Train the model
def train_model(model, train_loader, replay_buffer, criterion, optimizer, device, task_id, batch_size):
    model.train()
    total_correct, total_samples, total_loss = 0, 0, 0

    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    if task_id != 0:
        replay_data, replay_target = replay_buffer.get_replay_data(batch_size)
        replay_data, replay_target = replay_data.to(device), replay_target.to(device)
        replay_data = replay_data.unsqueeze(1)
        print(f"Replay Data Shape: {replay_data.shape}, Data Shape: {data.shape}")
        combined_data = torch.cat([data, replay_data], dim=0)
        combined_target = torch.cat([target, replay_target], dim=0)
    else:
        combined_data, combined_target = data, target

    indices = torch.randperm(combined_data.size(0))
    combined_data, combined_target = combined_data[indices], combined_target[indices]

    for i in range(0, combined_data.size(0), batch_size):
        mini_batch_data = combined_data[i:i + batch_size]
        mini_batch_target = combined_target[i:i + batch_size]
        outputs = model(mini_batch_data)
        loss = criterion(outputs, mini_batch_target).mean()  # Average the per-sample losses for a scalar loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_samples += mini_batch_target.size(0)
        total_correct += (predicted == mini_batch_target).sum().item()
        total_loss += loss.item() * mini_batch_data.size(0)
    
    avg_loss = total_loss / total_samples
    training_accuracy = 100 * total_correct / total_samples
    print(f"Training Accuracy: {training_accuracy:.2f}%")
    return model, avg_loss

# Main function for iCaRL
def main(params):
    dataset_name = params['dataset_name']
    num_tasks = params['num_tasks']
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    input_size = params['input_size']
    num_classes_per_task = params['num_classes_per_task']
    input_channels = params['input_channels']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(input_channels, num_classes_per_task * num_tasks, input_size).to(device)
    replay_buffer = ReplayBuffer(capacity=750)

    criterion = nn.CrossEntropyLoss(reduction='none')  # Ensuring criterion has reduction='none'
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    validation_accuracies = {}

    for task_id in range(num_tasks):
        train_dataset, _ = preprocess_images(dataset_name=dataset_name, task_id=task_id, train=True)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

        for epoch in range(num_epochs):
            model, avg_loss = train_model(model, train_loader, replay_buffer, criterion, optimizer, device, task_id, batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss: {avg_loss:.4f}")

        # Feature extraction and exemplar selection for replay
        features = extract_features(model, train_loader, device)
        loss_values = compute_loss_values(model, train_loader, criterion, device)
        targets = torch.cat([target for _, target in train_loader], dim=0)

        selector = KMeansExemplarSelector(features, targets, loss_values, max_retrieval=750)
        selected_indices = selector.select_exemplars(num_clusters=15)

        selected_dataset = Subset(train_dataset, selected_indices)
        exemplar_images, exemplar_labels = next(iter(DataLoader(selected_dataset, batch_size=len(selected_dataset), shuffle=False)))

        replay_buffer.add_exemplars(task_id, list(zip(exemplar_images, exemplar_labels)))

        # Validation
        validation_accuracies[task_id] = []
        for validate_task in range(task_id + 1):
            val_dataset, _ = preprocess_images(dataset_name=dataset_name, task_id=validate_task, train=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            validation_accuracy = evaluate_model(model, val_loader, device)
            validation_accuracies[task_id].append(validation_accuracy)
            print(f"Validation Accuracy for Task {validate_task + 1}: {validation_accuracy:.2f}%")

        print("All Validation Accuracies:", validation_accuracies)

    torch.save(model.state_dict(), 'trained_model_icarl.pth')

# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()
    return 100 * total_correct / total_samples

def compute_loss_values(model, data_loader, criterion, device):
    model.eval()
    loss_values = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            losses = criterion(outputs, target)  # Shape: (batch_size,)
            loss_values.extend(losses.cpu().numpy())
    return loss_values

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            features.append(model.features(data).cpu())
    return torch.cat(features, dim=0)

if __name__ == "__main__":
    params = {
        'dataset_name': 'MNIST',
        'num_tasks': 5,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'batch_size': 32,
        'input_size': (1, 28, 28),
        'num_classes_per_task': 2,
        'input_channels': 1
    }
    main(params)
