# Mask Detector

A two-part deep learning project that trains a face mask detection model and deploys it for real-time webcam inference. The training notebook builds and evaluates a **MobileNetV2-based** classifier, and the detection notebook runs live mask detection using **OpenCV's Haar Cascade** face detector combined with the trained model.

## Overview

The project is split into two notebooks: one for training the model on a labeled dataset of masked and unmasked faces, and another for real-time detection via webcam. The model uses transfer learning with **MobileNetV2** (pretrained on ImageNet) with a custom binary classification head for "With Mask" / "Without Mask" prediction.

## Project Structure

```
Mask-Detector-main/
├── Train_Mask_Detection.ipynb    # Model training notebook
└── Mask_Detector.ipynb           # Real-time webcam detection notebook
```

## Dataset

- **Structure required:**

```
dataset/
├── train/
│   ├── With Mask/       # Masked face images
│   └── Without Mask/    # Unmasked face images
└── val/
    ├── With Mask/
    └── Without Mask/
```

- **Image size:** resized to `224 × 224` for MobileNetV2 or `128 × 128` for the custom CNN path
- **Classes:** Binary — `With Mask (0)` / `Without Mask (1)`

---

## Notebook 1: Train\_Mask\_Detection.ipynb

### Workflow

#### 1. Data Loading & Preprocessing
- Read images from `With Mask` and `Without Mask` folders
- Convert BGR → RGB and resize to `128 × 128`
- Assign binary labels (0 = mask, 1 = no mask)
- Convert to NumPy arrays

#### 2. Data Augmentation (ImageDataGenerator)
- Rescale pixel values to `[0, 1]`
- Apply random rotation, zoom, width/height shifts, and horizontal flip
- Reserve 20% of data for validation via `validation_split`

#### 3. Model Architecture — MobileNetV2 (Transfer Learning)
```
MobileNetV2 (pretrained, base frozen)
  → Flatten
  → Dense(128, activation='relu')
  → Dropout
  → Dense(1, activation='sigmoid')   ← binary mask/no-mask output
```

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Input Size:** 224 × 224 × 3

#### 4. Training
- Fit model on `train_data` with validation on `val_data`
- Save trained model as **`mask_detector_model.h5`**

---

## Notebook 2: Mask\_Detector.ipynb

### Workflow

#### 1. Face Detection
- Load OpenCV's **Haar Cascade** frontal face detector (`haarcascade_frontalface_default.xml`)
- Detect faces in each webcam frame using `detectMultiScale`

#### 2. Mask Classification
- Crop detected face region from frame
- Resize to `224 × 224`, normalize, and expand dimensions
- Pass through loaded `mask_detector_model.h5`
- Classify as **With Mask** or **Without Mask** based on sigmoid output

#### 3. Real-Time Display
- Draw bounding boxes around detected faces
- Overlay classification label on each face
- Stream annotated frames in a live OpenCV window
- Press **`q`** to quit the webcam feed

---

## Requirements

- Python 3.x
- Jupyter Notebook or JupyterLab
- tensorflow / keras
- opencv-python
- numpy
- matplotlib
- scikit-learn
- imutils

Install all dependencies with:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn imutils jupyter
```

## Usage

### Step 1 — Train the model

```bash
jupyter notebook Train_Mask_Detection.ipynb
```

Update dataset paths in the notebook and run all cells. This saves `mask_detector_model.h5`.

### Step 2 — Run real-time detection

```bash
jupyter notebook Mask_Detector.ipynb
```

Ensure `mask_detector_model.h5` is in the same directory, then run all cells to start the webcam feed.

## Outputs

- **`mask_detector_model.h5`** — trained binary classification model
- **Training plots** — accuracy and loss curves over epochs
- **Live webcam feed** — real-time annotated video with face bounding boxes and mask/no-mask labels

## Key Concepts

| Concept | Purpose |
|---|---|
| MobileNetV2 | Lightweight pretrained CNN used as a feature extractor via transfer learning |
| Transfer Learning | Reuses ImageNet weights; only the custom head is trained from scratch |
| Haar Cascade | OpenCV's fast classical face detection algorithm for locating faces in frames |
| ImageDataGenerator | Applies augmentation on-the-fly to increase training data variety |
| Binary Cross-Entropy | Loss function for two-class (mask / no mask) classification |
| Real-Time Inference | Frame-by-frame webcam processing using OpenCV VideoCapture |
