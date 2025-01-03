# DiT Diffusion Model Implementation Documentation

This project implements an image generation model based on Diffusion Transformer (DiT), combining Transformer architecture with diffusion models for high-quality image generation tasks.

## Model Architecture

### Architecture Overview

![DiT Model Architecture](./assets/dit_graph.png)

The above figure shows the complete DiT model architecture, which mainly includes:
- Input Layer: Including image patch embeddings, time embeddings, and label embeddings
- Backbone Network: 3 DiT blocks, each containing self-attention layers and feed-forward networks
- Conditional Control: Using time and label information to modulate each DiT block
- Output Layer: Generates predicted noise or images

## Training Process

### Training Results

![Training Loss Curve](./assets/training_loss.png)

The training loss curve demonstrates the model's learning process:
- X-axis: Training epochs
- Y-axis: L1 loss value
- Blue line: Training loss for each batch
- Orange line: Average loss per epoch

From the loss curve, we can observe:
- The stability of the model training process
- The overall downward trend of loss values
- Convergence status

### Core Components

1. **DiT Model Structure**
   - Image size: 256×256 pixels
   - Patch size: 4×4
   - Number of channels: 3 (RGB)
   - Number of DiT blocks: 3
   - Number of attention heads: 6
   - Embedding dimension: 64
   - Number of labels: 1

2. **Time Embedding Module**
   - Frequency embedding dimension: 256
   - Uses sinusoidal position encoding
   - Includes MLP transformation layer

3. **DiT Block Structure**
   - Multi-head self-attention mechanism
   - Feed-forward neural network
   - Layer normalization
   - Adaptive layer parameters (α, β, γ)

## Training Process

### Dataset
- Custom panda dataset
- Images preprocessed and normalized to [-1, 1] range
- Uses persistent workers for data loading (num_workers=10)

### Training Configuration
- Optimizer: Adam
- Learning rate: 1e-6
- Loss function: L1 loss
- Batch size: Configured through `config.py`
- Diffusion steps: Defined by `max_t` in `config.py`
- Running device: Supports GPU (CUDA)

### Diffusion Process
- β value scheduling: Linear increase from 0.0001 to 0.02
- Forward process: Gradually adds Gaussian noise
- Reverse process: Iterative denoising
- Uses pre-computed values for variance scheduling

### Monitoring and Visualization
- Integrated TensorBoard for training monitoring
- Tracked metrics include:
  - Loss per batch
  - Average loss per epoch
  - Noise prediction histograms
  - Generated sample images
- Model architecture visualization through TensorBoard

### Model Saving
- Regular saving of model states
- Supports continuing training from checkpoints
- Configurable checkpoint paths

## Training Features

1. **Progressive Training**
   - Supports incremental training
   - Automatically loads existing checkpoints

2. **Visualization Features**
   - Regularly generates sample images
   - Visualizes training progress
   - Tracks batch-level and epoch-level metrics

3. **Performance Optimization**
   - Persistent workers improve data loading efficiency
   - GPU acceleration support
   - Configurable batch size and number of workers

## Usage Instructions

1. **Environment Setup**
   ```bash
   # Required packages
   torch
   torchvision
   tensorboard
   tqdm
   ```

2. **Start Training**
   ```python
   python train.py
   ```

3. **Monitor Training**
   ```bash
   tensorboard --logdir=./dit_training
   ```

## Model Output

The model generates images through an iterative denoising process:
- Starts from random noise
- Refines the image through multiple iterations
- Final output is constrained to [-1, 1] range
- Supports label-conditioned generation

## Implementation Details

- Uses custom image patch embedding layer for patch processing
- Position embeddings are learnable parameters
- Diffusion process uses pre-computed scheduling for efficiency
- Training process includes gradient clipping for stability
- Regular visualization to monitor training progress

## Custom Configuration

Model architecture and training process can be customized through:
- `config.py`: General settings
- Model initialization parameters in `train.py`
- Dataset configuration in `dataset.py`

## Important Notes

1. **Hardware Requirements**
   - GPU recommended for training
   - Sufficient VRAM needed for chosen batch size

2. **Training Tips**
   - Recommended to test on small datasets first
   - Regularly check generated sample quality
   - Adjust learning rate and batch size based on actual conditions

3. **Troubleshooting**
   - If memory issues occur, reduce batch size
   - If training is unstable, adjust learning rate or increase gradient clipping
   - Ensure correct data preprocessing and image normalization