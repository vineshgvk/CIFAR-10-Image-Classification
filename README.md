
# CIFAR-10 Image Classification

This project classifies images from the CIFAR-10 dataset using a convolutional neural network (CNN) and support vector machine (SVM) model.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images belonging to 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The dataset is divided into 50,000 training images and 10,000 test images.

The images are preprocessed by:

- Normalizing the pixel values between 0-1
- Flattening the label arrays into 1D
- Rotating and flipping the images for data augmentation


## Technologies

- Python 3.6
- TensorFlow 2.0
- Keras 
- scikit-learn
- Matplotlib
- NumPy
- OpenCV


## Models

Two models are implemented:

### Convolutional Neural Network

A CNN model with the following architecture:

- Convolutional layers: 32, 64, 128 filters with 3x3 kernels 
- Max pooling layers
- Flatten layer
- Fully connected layer with 1024 nodes
- Output layer with 10 nodes and softmax activation

The model is compiled with categorical cross-entropy loss and adam optimizer.

### Support Vector Machine

An SVM model with polynomial and RBF kernels trained on grayscale pixel intensities of images.

SVM hyperparameters are tuned using grid search cross-validation. 

## Usage

The main steps are:

1. Load and preprocess CIFAR-10 dataset
2. Build CNN model
3. Train CNN model
4. Evaluate CNN model on test set
5. Convert images to grayscale
6. Build SVM model with polynomial and RBF kernels 
7. Train SVM models
8. Evaluate SVM models on test set
9. Compare performance of models

## Results

The CNN model achieves ~85% accuracy on the test set. The SVM models achieve ~45% accuracy.

The CNN significantly outperforms SVM, indicating that CNNs are better suited for visual classification tasks compared to SVMs.

## References

CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
