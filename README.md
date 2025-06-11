# Sobel Module PyTorch Implementation

This is a PyTorch implementation of a Sobel edge detection module combined with CNN and Transformer architectures for image classification tasks. The project is specifically optimized for the CIFAR-10 dataset.

## Features

- Implementation of learnable Sobel edge detection layer
- Integration with ResNet34 as backbone network
- Incorporation of Transformer architecture for feature extraction
- Support for mixed precision training
- Comprehensive data augmentation strategies
- Support for training checkpoint resumption

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/sobel_module_pytorch.git
cd sobel_module_pytorch
```

2. Install dependencies:
```bash
pip install torch torchvision
```

## Usage

### Training the Model

Run the training script directly:
```bash
python train_cifar10.py
```

Model weights will be automatically saved during training with the format `cnn_transformer_epoch{epoch}.pt`.

### Model Architecture

The model consists of the following components:
- StructuredSobelLayer: Learnable Sobel edge detection layer
- ResNet34 backbone (modified version)
- Transformer encoder
- Classification head

### Data Augmentation

The following data augmentation strategies are used during training:
- Random cropping
- Horizontal flipping
- Random rotation
- Color jittering
- Cutout augmentation

## Training Logs

Training logs are saved in the `train.log` file, including:
- Training loss and accuracy for each epoch
- Test set loss and accuracy
- Learning rate changes

## Notes

- Ensure sufficient GPU memory (if using GPU for training)
- Training process may take a long time, checkpoint resumption is recommended
- Hyperparameters in `train_cifar10.py` can be adjusted as needed

## License

MIT License 
