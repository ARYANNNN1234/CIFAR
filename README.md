# CIFAR-10 Image Classification with CNN and Interpretability

## Overview
This project implements a Convolutional Neural Network (CNN) for multi-class image classification on the CIFAR-10 dataset. Beyond standard model training and evaluation, it delves into model interpretability using advanced techniques like Saliency Maps and GradCAM to understand how the CNN makes its predictions. The project also explores the impact of various architectural choices during experimentation.

---

## Dataset
- **Dataset:** CIFAR-10 (60,000 32x32 color images, 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Structure:**
  - Images: `.png` files
  - Labels: `cifar10Labels.csv` maps image indices to labels

You can download the dataset directly:
- [cifar10.zip](https://infyspringboard.onwingspan.com/common-content-store/Shared/Shared/Public/lex_auth_012782825259556864334_shared/web-hosted/assets/cifar10.zip)
  - Place this file in your project directory and extract its contents.

---

## Methodology

### 1. Data Loading and Preprocessing
- Load images from directory, resize to 32x32 pixels.
- Normalize pixel values to `[0, 1]`.
- Encode labels with `LabelEncoder` and one-hot encode using `keras.utils.to_categorical`.
- Split into training and testing sets.

### 2. CNN Architecture
- **Model:** Sequential Keras model
  - **Input Layer:** 32x32x3
  - **Convolutional Blocks:** Two `Conv2D` layers (filters: 32, 64; kernel: 3x3; activation: relu; L2 regularization)
    - Each followed by `BatchNormalization` and `MaxPool2D` (2x2)
  - **Flatten Layer:** Converts feature maps to a 1D vector
  - **Output Layer:** Dense (10 units, softmax)

### 3. Model Training and Evaluation
- Compile with `categorical_crossentropy` loss and Adam optimizer.
- Train for 5 epochs, 20% validation split.
- Use `ModelCheckpoint` to save the best model (based on validation accuracy).
- Evaluate accuracy on train and test sets.
- Visualize confusion matrices for both sets.

### 4. Model Interpretability (Explainable AI - XAI)
- Use `tf-keras-vis` for Saliency Maps and GradCAM:
  - **Saliency Maps:** Visualize pixels most influencing predictions.
  - **GradCAM:** Overlay heatmaps on images to highlight important regions for class prediction.
- Temporarily apply linear activation to the last model layer for gradient-based interpretability.

### 5. Experimentation
- **Varying Strides:** Test strides of 1, 2, 3 in Conv2D layers.
- **Different Kernel Sizes:** Experiment with 3x3 and 4x4 kernels.

---

## How to Run the Project

### 1. Dataset Setup

```
your_project_folder/
├── cifar10Labels.csv
└── cifar10/
    └── cifar10/
        ├── 0.png
        ├── 1.png
        └── ...
├── cifar10_cnn_project.py
└── CIFAR10_checkpoint.keras    # Generated after first run
├── cifar10.zip                # Download and extract this file
```

### 2. Install Dependencies

```sh
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras imageio Pillow tf-keras-vis
```

### 3. Run the Script

```sh
python cifar10_cnn_project.py
```

This will:
- Load and preprocess the CIFAR-10 data
- Train the CNN model
- Save the best model checkpoint (`CIFAR10_checkpoint.keras`)
- Evaluate model accuracy and visualize confusion matrices
- Generate and display interpretability maps (Saliency, GradCAM)

---

## Files in this Project
- `cifar10_cnn_project.py`: Main Python script (model, training, evaluation, interpretability)
- `cifar10Labels.csv`: CIFAR-10 image labels
- `cifar10/cifar10/`: CIFAR-10 image files
- `CIFAR10_checkpoint.keras`: Saved best model checkpoint
- `out.csv`: (Optional) Predictions on the test set
- `cifar10.zip`: Downloadable dataset file

---

## Future Enhancements
- Implement deeper CNN architectures (ResNet, Inception, EfficientNet)
- Integrate advanced data augmentation (rotations, shifts, flips)
- Explore transfer learning with pre-trained models on CIFAR-10

---

## License
Distributed under the MIT License. See `LICENSE` for details.

---

## Contact
For questions or suggestions, please open an issue or contact [repo owner](https://github.com/ARYANNNN1234).
