# Diffusion Models - Pytorch

Diffusion models are primarily inspired from non-equilibrium thermodynamics and they are used to generate samples from the data distribution using a parameterised Markov chain trained using variational inference. ([Ho et al., 2020](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)) experimentally proved that we can generate high quality images using diffusion probabilistic models which were previously introduced in ([Sohl-Dickstein et al., 2015](https://arxiv.org/pdf/1503.03585.pdf)), allowing these models to compete with other generative models in terms of sample quality.

The goal of this project is to get familiar with the Diffusion Models literature, learn about the advancements in the field, the techniques that result in its overall success and get hands-on experience implementing the pipeline. This repository is based on the implementations of (Niels Rogge, 2022) [https://huggingface.co/blog/annotated-diffusion](https://huggingface.co/blog/annotated-diffusion), (Wang, 2020) [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) and (Ho, 2019) [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) as references, that can generate good quality images from the data distribution and captures frequently used methods in the field. **Please note that I've tried to modularize the previous implementations to make it easier to understand and follow and build on top of and there are references added within the code.**

Given the computational limitations, I mainly used MNIST, FashionMNIST and CIFAR10 in my experiments. But it's straight forwarded to use the code for other datasets as well.

I analysed the effect of parameters such as length of the markov chain T , variance initialization of $\beta_t$ and different loss functions have on the images generated and the training process which helps us understand what techniques help improve the quality of the generated images. It was observed that increasing the length of the Markov Chain (T) results in improved quality of images. The model is fairly robust to different initializations of variance parameters $\beta_t$ and the modern scheduling techniques didn’t seem to have much impact in the image quality. 

The weighted L2 loss function which aligns with the theory to gave the best results on FashionMNIST contrary to the (Ho et al., 2020). But This loss function was found to be highly unstable during training, resulting in NaN values and inorder to stabilize the training, we clip the weights. From the experimental analysis, L2 weighted loss with weights clipped between (0,1) gave the best results.

# Dependencies
If on google colab, or you have pip installed, install these packages using
```
pip install einops
pip install datasets
pip install torch-fidelity
pip install torchmetrics
```

# Getting started

Clone the repository and move into the main directory using, 
```
git clone https://github.com/GangaMegha/Diffusion-Models.git 
cd /Diffusion-Models
```

If training the model on MNIST, FashionMNIST or CIFAR10, you can use,
```
make train_mnist
make train_fashion_mnist
make train_cifar10
```

At each epoch, we sample generated images and they are stored in `/results` folder. The models are saved in `/checkpoint` and once the training is over, the log file with loss values at each epoch is saved in `/log`. You can change these locations in the `/src/config.py` file.

The code for training a vanilla CNN classifier to compute local metrics for FID and IS is available at `/examples/FashionMNIST_CNN_Classifier.ipynb`


