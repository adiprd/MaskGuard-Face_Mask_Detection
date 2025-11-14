# 🎭 MaskGuard AI - Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

**MaskGuard AI** is an intelligent computer vision system that automatically detects face masks and classifies them into three categories with high accuracy. Built with Transfer Learning using MobileNetV2, this system helps enforce safety protocols in public spaces.

## Features

- **Triple Classification**: Detects three mask conditions:
  - **With Mask** - Properly wearing mask
  - **Mask Weared Incorrect** - Improper mask usage
  - **Without Mask** - No mask detected

- ** High Accuracy**: Achieves over 95% accuracy on validation sets
- ** Real-time Ready**: Optimized for real-time deployment
- ** Mobile-Friendly**: Uses lightweight MobileNetV2 architecture
- ** Data Augmentation**: Enhanced training with image transformations

## 📊 Model Architecture
Input (224x224x3)
↓
MobileNetV2 (Base)
↓
Global Average Pooling
↓
Dense (128, ReLU)
↓
Output (3, Softmax) → [Incorrect, With_Mask, Without_Mask]


## Installation

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.0+
OpenCV
NumPy
Matplotlib
Scikit-learn
Clone Repository
bash
git clone https://github.com/yourusername/maskguard-ai.git
cd maskguard-ai
Install Dependencies
bash
pip install -r requirements.txt

Dataset Structure
The dataset should be organized as follows:

text
Dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── without_mask/
│   ├── image1.jpg
│   └── ...
└── mask_weared_incorrect/
    ├── image1.jpg
    └── ...

Usage
1. Training the Model
python
from maskguard import MaskDetector

# Initialize detector
detector = MaskDetector()

# Train model
history = detector.train(
    dataset_path="/path/to/dataset",
    epochs=10,
    validation_split=0.2
)
2. Making Predictions
python
# Load trained model
model = detector.load_model("face_mask_model.h5")

# Predict single image
result = detector.predict_image("test_image.jpg")
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.4f}")
3. Real-time Detection
python
# Start webcam detection
detector.start_webcam()

Performance Metrics
Classification Report
text
                      precision    recall  f1-score   support

mask_weared_incorrect       0.96      0.94      0.95       320
           with_mask       0.98      0.97      0.98       400
        without_mask       0.95      0.96      0.95       350

            accuracy                           0.96      1070
           macro avg       0.96      0.96      0.96      1070
        weighted avg       0.96      0.96      0.96      1070

Training History
https://images/training_history.png

Confusion Matrix
https://images/confusion_matrix.png

Demo
Single Image Prediction
bash
python demo.py --image test_image.jpg
Webcam Real-time Detection
bash
python webcam_demo.py
Batch Processing
bash
python batch_process.py --input_folder images/ --output_folder results/

Project Structure
text
maskguard-ai/
├── models/
│   ├── face_mask_model.h5
│   └── mobileNetV2_base/
├── src/
│   ├── __init__.py
│   ├── mask_detector.py
│   ├── train.py
│   └── utils.py
├── datasets/
│   ├── train/
│   └── validation/
├── demos/
│   ├── single_image.py
│   └── webcam_demo.py
├── requirements.txt
├── train_model.ipynb
└── README.md

Customization
Adding New Classes
python
# Modify the output layer for new classes
model = MaskDetector(num_classes=4)  # Add new class
Changing Base Model
python
# Use different pre-trained models
detector = MaskDetector(base_model='ResNet50')

Applications
Healthcare Facilities: Monitor mask compliance in hospitals
Office Buildings: Ensure workplace safety protocols
Retail Stores: Automated entry control
Educational Institutions: Campus safety monitoring
Public Transportation: Mass transit safety enforcement
Contributing

We welcome contributions! Please see our Contributing Guide for details.

Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
MobileNetV2 by Google Research
TensorFlow & Keras teams
Dataset providers and contributors
OpenCV community for computer vision tools
