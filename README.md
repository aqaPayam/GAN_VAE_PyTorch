
# GAN and VAE Implementation - Deep Learning Course by Dr. Soleymani

## Repository Overview
This repository contains the implementation of two popular generative models: **Generative Adversarial Networks (GAN)** and **Variational AutoEncoders (VAE)**. These implementations are part of a homework assignment for Dr. Soleymani's deep learning course. The models are trained on the MNIST dataset, which consists of grayscale images of handwritten digits.

The notebook walks through both model architectures, their training processes, and evaluation techniques for generating new image data.

### Structure
- **`GAN_VAE_Implementation.ipynb`**: Jupyter notebook containing the complete implementation of the VAE and GAN models, along with explanations and code for training and generating new data.

### Prerequisites
To run the code, you need to have the following installed:
- Python 3.x
- PyTorch (for model building and training)
- NumPy (for numerical computations)
- Matplotlib (for plotting the generated images)
- TQDM (for monitoring the progress of training loops)

To install the required packages, you can use:
```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Detailed Explanation of Models

### 1. Variational AutoEncoder (VAE)
The VAE is a type of autoencoder that learns a probabilistic latent space. Unlike standard autoencoders that compress and decompress data in a deterministic way, VAEs assume a latent distribution and train the model to sample from this distribution to reconstruct the input data.

- **Encoder**: Maps the input data to a latent space by predicting the mean and variance of a Gaussian distribution.
- **Latent Space Sampling**: From the predicted mean and variance, a latent vector is sampled using the reparameterization trick, which allows gradients to flow through stochastic sampling.
- **Decoder**: Takes the latent vector and reconstructs the original input.

The loss function is a combination of:
- **Reconstruction Loss**: Measures how well the decoder reconstructs the input.
- **KL Divergence**: Encourages the learned latent distribution to be close to a standard normal distribution.

The goal of the VAE is to learn a smooth latent space where small changes in the latent vector result in smooth changes in the generated output.

### 2. Generative Adversarial Network (GAN)
The GAN is composed of two networks: the **Generator** and the **Discriminator**. These two networks compete in a zero-sum game where the generator tries to fool the discriminator with fake data, while the discriminator tries to distinguish between real and fake data.

- **Generator**: Takes random noise (from a standard normal distribution) as input and generates fake images.
- **Discriminator**: Takes an image (either real or generated) as input and predicts whether it is real or fake.

Training a GAN involves two loss functions:
- **Generator Loss**: Encourages the generator to produce images that the discriminator classifies as real.
- **Discriminator Loss**: Encourages the discriminator to correctly classify real images as real and fake images as fake.

The models are trained iteratively, alternating between optimizing the discriminator and the generator.

## Notebook Sections

### 1. **Data Loading**
   - The notebook begins by loading the **MNIST dataset**, which contains 28x28 grayscale images of handwritten digits (0-9).
   - Data is split into training and validation sets, and PyTorch’s `DataLoader` is used to efficiently load batches of data during training.

### 2. **Model Architecture**
   - The VAE and GAN models are implemented in PyTorch. The architecture of each model is fully explained and implemented in a modular fashion.
   
   - **VAE Architecture**: 
     - Encoder: Consists of convolutional layers followed by fully connected layers, outputting the mean and log variance of the latent space.
     - Decoder: Mirrors the encoder and consists of fully connected layers followed by deconvolutional layers to reconstruct the original image.
   
   - **GAN Architecture**:
     - Generator: Contains several layers of up-sampling and transposed convolutions to generate an image from random noise.
     - Discriminator: A series of convolutional layers that down-sample the image and classify it as real or fake.

### 3. **Training**
   - The training loops for both the VAE and GAN are implemented with detailed explanations of each step.
   - **VAE Training**: The VAE is trained by minimizing the combined reconstruction loss and KL divergence. After each epoch, the decoder is used to generate and visualize new images by sampling from the latent space.
   
   - **GAN Training**: The GAN is trained using alternating updates to the generator and discriminator. The generator aims to create realistic images that the discriminator classifies as real, while the discriminator learns to better distinguish real images from fakes.

### 4. **Evaluation**
   - After training, the models are evaluated based on their ability to generate realistic images.
   - **VAE Evaluation**: The latent space is sampled to generate new images. Visualizations of these generated images are included in the notebook to show the quality of the VAE’s output.
   - **GAN Evaluation**: Similar to the VAE, the GAN’s generator is used to produce new images. The notebook shows samples of the images generated during training and provides a qualitative analysis of their realism.

### 5. **Results**
   - The final section of the notebook displays the results of both the VAE and GAN. Images generated by the VAE are compared against those generated by the GAN, with insights into the advantages and limitations of each model.

### Additional Details
- The notebook includes interactive visualizations of the latent space and the images generated by both models.
- Comments are provided throughout the code to explain the purpose of each function and the logic behind each step in the training loop.

## How to Run the Notebook
1. Clone the repository:
```bash
git clone https://github.com/AqaPayam/GAN_VAE_PyTorch.git
```

2. Navigate to the project directory:
```bash
cd GAN_VAE_PyTorch
```

3. Run the Jupyter notebook:
```bash
jupyter notebook GAN_VAE_Implementation.ipynb
```

4. Train the models and generate images by executing the cells in the notebook.

## Results
By running the notebook, you will:
- Train a VAE to generate new images that resemble the digits in the MNIST dataset by sampling from the latent space.
- Train a GAN to generate high-quality images of handwritten digits by competing the generator and discriminator networks.

Both the VAE and GAN models will generate images that are visualized throughout the notebook, showcasing their performance.

## Contribution
Contributions are welcome! You can improve the models, experiment with different datasets, or extend the implementations to more complex architectures. Feel free to fork the repository and create a pull request with your improvements.

## License
This project is licensed under the MIT License.
