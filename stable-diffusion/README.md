
# Methodology

## Perceptual Image Compression

- Autoencoder

- Loss function : Perceptual loss + Patch-based adversarial objective

Preserves image likeness while punishing blurring

- Avoid high-variance latent space, so KL regularization + VQ regularization (aka vector quantization in the decoder)


## Latent Diffusion Models

### Diffusion models 


- learn distribution through gradual denoising of a normal distribution

- aka learning reverse process of [Markov chain](https://brilliant.org/wiki/markov-chains/#:~:text=A%20Markov%20chain%20is%20a,possible%20future%20states%20are%20fixed.)

- In image processing, use variational lower bound (?) like denoising score-matching (?)


### Generative Modeling of Latent Representations

- Cut away the fluff in image space to work with the core latent space (better for distributive modeling)


- Originally, it ran an attention model purely in the latent space

- However, with generative modeling/diffusion models, can take advantage of "inductive biases"

- Loss is calculated by the expectation of the difference between the original layer output and the denoised decoder output at timestep t.

## Conditioning Mechanism

- Basically, the UNet has attention in its layers. Specifically, cross-attention, because the input of image latent space and CLIP text encoding space is different

- Transformer background (?)

- The Q value is the input from the latent space of image

- K and V are derived from the CLIP text encoding



## Moving to Latent Space

- Perceptual Compression : Fewer pixels, but same image

AutoEncoder + GAN

- Semantic Compression : Image details lost, but same idea 

Latent Diffusion Model


## Overview of architectures

![diagram](diagram.jpeg)

- UNet?
- CLIP ViT-L/14 Text encoder

## Data

- LAION database
- 
