# Cyberbullying Detection in Bangla-English Social Media Comments

This repository provides a complete pipeline for detecting cyberbullying in Bangla-English social media comments using deep learning and natural language processing. It includes two main Jupyter notebooks:

---

## 1. CyberbullyingDetect.ipynb

This notebook demonstrates the end-to-end workflow for single-model (BERT-based) cyberbullying detection:

- **Data Loading & Exploration:**  
  Loads a Bangla-English comments dataset and explores its structure, label distribution, and comment lengths.

- **Text Preprocessing:**  
  Cleans and normalizes comments by removing URLs, punctuation, numbers, emojis, and both Bangla and English stopwords.

- **Visualization:**  
  Plots the distribution of comment categories and comment lengths to understand dataset characteristics and potential class imbalance.

- **Feature Extraction:**  
  Uses a multilingual BERT model to encode cleaned texts and extract sentence embeddings.

- **Model Building & Training:**  
  Defines and trains a custom BERT-based classifier for multi-class cyberbullying detection, using PyTorch.

- **Evaluation:**  
  Reports accuracy, classification metrics, confusion matrix heatmaps, and ROC curves for each class.

- **Prediction & Saving:**  
  Provides a function to predict the category of new comments and saves the trained model, tokenizer, label encoder, and preprocessed data for future use.

---

## 2. EnsembleModel.ipynb

This notebook extends the pipeline with an ensemble approach for improved performance:

- **Base Models:**  
  Trains three base classifiers: BERT (deep learning transformer), LSTM (recurrent neural network), and Random Forest (traditional machine learning).

- **Stacked Ensemble:**  
  Combines the predictions of all base models using a logistic regression meta-learner, leveraging their probability outputs as meta-features.

- **Evaluation & Visualization:**  
  Compares the ensemble and base models using accuracy, weighted F1-score, classification reports, confusion matrices, ROC curves, and precision-recall curves.  
  Visualizes class distribution in the test set and provides detailed performance plots.

- **Prediction Function:**  
  Offers a unified prediction function that outputs the ensemble and base model predictions for any new comment.

- **Model Saving:**  
  Saves the meta-learner (ensemble model) for future inference.

---

## How to Use

1. **Run `CyberbullyingDetect.ipynb`** to preprocess data, train a BERT classifier, and evaluate single-model performance.
2. **Run `EnsembleModel.ipynb`** to train base models, build the stacked ensemble, and compare results.
3. Use the provided prediction functions to classify new comments.
4. Saved models and encoders can be loaded for deployment or further analysis.

---

## Visualizations Included

- Label distribution bar plots
- Comment length histograms
- Confusion matrix heatmaps (for all models)
- ROC curves and AUC scores (per class)
- Precision-recall curves
- Accuracy and F1-score comparison bar plots
- Test set class distribution

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- pandas, numpy, matplotlib, seaborn, nltk

---

## Citation

If you use this code or dataset, please cite the corresponding paper or repository.

---

For questions or contributions, please open an issue