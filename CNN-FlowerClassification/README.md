# Flower Classification using Convolutional Neural Networks

A deep learning project for automated flower classification using transfer learning with VGG16 architecture. This model classifies flowers into five distinct categories with high accuracy.

**Author:** Eimaan Khan  
**Institution:** UET Mardan - BS in Computer Science (Specialization in AI)

---

## Project Overview

This project implements a Convolutional Neural Network (CNN) for classifying flower images into five categories: daisy, dandelion, rose, sunflower, and tulip. The model utilizes transfer learning with a pre-trained VGG16 architecture to achieve efficient and accurate results.

---

## Dataset

- **Source:** Custom flower image dataset
- **Classes:** 5 flower types (daisy, dandelion, rose, sunflower, tulip)
- **Structure:** 
  - Training set
  - Validation set
  - Test set
- **Image Resolution:** 224 × 224 pixels

---

## Model Architecture

### Approach: Transfer Learning

- **Base Model:** VGG16 (pre-trained on ImageNet weights)
- **Fine-tuning Strategy:** Bottleneck feature extraction with custom fully-connected layers

### Top Model Architecture

```
Flatten Layer
    ↓
Dense(100) + LeakyReLU(α=0.3)
    ↓
Dropout(0.5)
    ↓
Dense(50) + LeakyReLU(α=0.3)
    ↓
Dropout(0.3)
    ↓
Dense(5) + Softmax (Output Layer)
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | RMSprop (lr=1e-4) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 5 |
| Epochs | 7 |
| Validation Strategy | Separate validation set |

---

## Results

- **Training/Validation Accuracy:** ~95%+
- **Test Accuracy:** High performance across all flower classes
- **Evaluation Metrics:**
  - Classification Report (per-class precision, recall, F1-score)
  - Confusion Matrix (normalized and non-normalized)
  - Per-class performance analysis

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Deep Learning** | Keras, TensorFlow |
| **Data Processing** | NumPy, Pandas |
| **Image Processing** | Keras ImageDataGenerator, PIL/Pillow |
| **Visualization** | Matplotlib |
| **Metrics & Analysis** | scikit-learn |
| **Language** | Python 3.x |

---

## Features

✓ Transfer learning with VGG16 for efficient training  
✓ Bottleneck feature extraction (pre-computed for faster iterations)  
✓ Data augmentation through ImageDataGenerator  
✓ Comprehensive model evaluation with confusion matrix  
✓ Single image prediction with confidence scores  
✓ Visualized training/validation metrics  

---

## Usage

### Training

The notebook provides a complete pipeline:

1. **Feature Extraction:** VGG16 converts all images to bottleneck features (saved as .npy files)
2. **Model Training:** Sequential model trained on extracted features
3. **Evaluation:** Testing on validation and test sets

### Single Image Prediction

```python
path = 'data/test/sample_flower.jpg'
test_single_image(path)
```

This function:
- Loads and preprocesses the image
- Generates predictions with confidence scores
- Outputs the classified flower type

---

## Project Structure

```
flower_classification_cnn/
├── flower_classification_cnn.ipynb   # Main notebook
├── data/
│   ├── train/                        # Training images (by class)
│   ├── validation/                   # Validation images (by class)
│   └── test/                         # Test images (by class)
├── bottleneck_features_train.npy     # Extracted training features
├── bottleneck_features_validation.npy # Extracted validation features
├── bottleneck_features_test.npy      # Extracted test features
└── bottleneck_fc_model.h5            # Trained model weights
```

---

## Key Insights

- **Transfer Learning Efficiency:** By leveraging VGG16's pre-trained weights, the model achieves high accuracy with minimal training time
- **Dropout Regularization:** Used to prevent overfitting (0.5 after first dense layer, 0.3 after second)
- **LeakyReLU Activation:** Maintains gradient flow better than standard ReLU during backpropagation
- **Balanced Performance:** Model performs consistently across all five flower classes

---

## Performance Metrics

The model provides:
- Per-class precision, recall, and F1-scores
- Confusion matrix for error analysis
- Training/validation loss and accuracy curves
- Real-time predictions with confidence percentages

---

## Requirements

- Python 3.x
- keras
- tensorflow
- numpy
- pandas
- scikit-learn
- matplotlib
- Pillow (PIL)

---

## Acknowledgments

This project demonstrates practical application of deep learning and transfer learning techniques for image classification tasks, implemented as part of the AI specialization coursework at UET Mardan.

