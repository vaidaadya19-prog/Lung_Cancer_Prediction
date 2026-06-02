# Lung Cancer Malignancy Prediction System

## Overview

This project presents an end-to-end Artificial Intelligence system for predicting lung nodule malignancy using a dual-pipeline architecture that combines traditional machine learning and deep learning approaches. The system was developed using the DICOM-LIDC-IDRI-Nodules dataset and deployed as a publicly accessible web application for real-time inference.

The objective of the project is to support early lung cancer detection by providing accurate, interpretable, and accessible malignancy predictions from radiological data.

## Problem Statement

Early identification of malignant pulmonary nodules remains a critical challenge in radiology. Manual assessment is time-consuming and subject to inter-observer variability. This project addresses the problem by developing an AI-assisted decision support system capable of classifying nodules as benign or malignant while providing explainable predictions.

## Solution Architecture

The system consists of two independent prediction pipelines:

### Tabular Machine Learning Pipeline

* Extracts radiomic features from DICOM Structured Report (SR) files.
* Implements Random Forest, Gradient Boosting, and Logistic Regression models.
* Evaluates multiple class imbalance handling strategies including SMOTE, BorderlineSMOTE, and CTGAN.
* Provides model interpretability through SHAP-based feature importance analysis.

### Deep Learning Image Pipeline

* Processes lung nodule segmentation images extracted from DICOM SEG files.
* Utilizes a fine-tuned ResNet18 convolutional neural network.
* Generates visual explanations using Grad-CAM.

Both pipelines are integrated into a unified Streamlit application for real-time prediction and visualization.

## Technical Stack

**Programming Language:** Python

**Machine Learning:** Scikit-learn, Imbalanced-Learn

**Deep Learning:** PyTorch, Torchvision

**Medical Imaging:** Pydicom, OpenCV

**Explainable AI:** SHAP, Grad-CAM

**Synthetic Data Generation:** CTGAN (SDV)

**Deployment:** Streamlit Cloud

## Key Results

| Metric   | Best Tabular Model | ResNet18 CNN |
| -------- | ------------------ | ------------ |
| AUC-ROC  | 0.880              | 0.740        |
| Recall   | 0.859              | 0.297        |
| F1-Score | 0.708              | 0.395        |

Key findings:

* Gradient Boosting achieved the highest overall AUC-ROC performance.
* BorderlineSMOTE significantly improved malignant case detection.
* SHAP analysis identified clinically relevant radiomic features driving predictions.
* The tabular pipeline outperformed the image pipeline due to the limited information available in segmentation masks compared to full CT scans.

## Highlights

* Designed and implemented a complete medical AI workflow from data processing to deployment.
* Built and compared multiple machine learning and deep learning models.
* Addressed real-world class imbalance challenges using advanced oversampling techniques.
* Integrated Explainable AI methods to improve model transparency.
* Deployed a production-ready web application for real-time prediction.

## Future Improvements

* Train CNN models directly on CT intensity images instead of segmentation masks.
* Develop 3D convolutional neural networks for volumetric analysis.
* Implement ensemble fusion of tabular and image-based predictions.
* Add uncertainty estimation and confidence calibration.
* Extend evaluation using additional public lung cancer datasets.

## Repository Structure

```text
├── data/
├── notebooks/
├── models/
├── app.py
├── requirements.txt
└── README.md
```

## Live Application

The deployed application enables users to:

* Submit radiomic feature values for prediction.
* Upload lung nodule images for classification.
* View confidence scores and explainability outputs.
* Compare predictions across different input modalities.

