# Replay Buffer Compression for Continual Learning

This repository contains the implementation for my research on compressing replay buffers in Continual Learning (CL) using different compression methods. The goal is to reduce memory usage while retaining task-relevant information, thus mitigating catastrophic forgetting.

## Compression Methods

The following compression methods are implemented:

1. **JPEG** - A standard lossy image compression technique.
2. **K-SVD** - A dictionary learning algorithm that performs sparse coding.
3. **PCA** - Principal Component Analysis, which reduces dimensionality by keeping the most significant components.
4. **CSC (Convolutional Sparse Coding)** - A feature-based method that learns sparse filters to represent data.

Currently, the compression methods have been tested on the **MNIST** dataset. Future work will involve extending this to datasets with color images like **CIFAR-10** and other challenging benchmarks.

## Running the Code

1. Clone this repository:

   ```bash
   git clone https://github.com/AlEXSH3N/Compressed-Replay-Method.git
   cd Compressed-Replay-Method
   ```

2. Navigate to the `train_model.py` file, where the training process and configurations are defined.

3. Select the compression method and sampling strategy within the `params` section under the `if __name__ == "__main__":` part of the script:

   ```python
   params = {
       ...
       'compression_method': 'JPEG',  # Options: 'JPEG', 'K-SVD', 'PCA', 'CSC'
       'sampling_method': 'DPP',      # Options: None, 'K-means', 'DPP'
       ...
   }
   ```

4. Once you've set your desired configuration, run the script:
   ```bash
   python train_model.py
   ```

## Dependencies

Make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```
