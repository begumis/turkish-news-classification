# Turkish News Classification (TTC4900 Dataset)

This project focuses on classifying Turkish news texts using various machine learning and deep learning techniques.

## 📌 Project Summary

This project aims to classify Turkish news texts from the [TTC4900 dataset](https://www.kaggle.com/datasets/savasy/ttc4900) into one of seven categories:

- Politics
- World
- Economy
- Culture
- Health
- Sports
- Technology

---

## ⚙️ Methodology

### 1. Preprocessing
- Lowercasing
- Removing special characters
- Stopword removal
- Tokenization
- Lemmatization

### 2. Feature Extraction
- Bag of Words (BoW)
- TF-IDF
- Word2Vec

### 3. Classification Models

#### Traditional ML Models
- Logistic Regression
- XGBoost
- Decision Tree
- Random Forest
- K-Nearest Neighbors (K-NN)
- LightGBM

#### Deep Learning Models
- Artificial Neural Networks (ANN)
- CNN, LSTM, BiLSTM, GRU
- Random & Word2Vec Embeddings

#### Transformer-based Models
- BERT Multilingual (bert-base-multilingual-uncased)
- Fine-tuned BERT for Turkish (`savasy/bert-turkish-text-classification`)

---

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 🏆 Key Results

- **Best ML model:** TF-IDF + Logistic Regression (91% Accuracy)
- **Best Deep Learning result:** Random Embeddings > Word2Vec Embeddings
- **Best overall performance:** Fine-tuned Turkish BERT model

---

## 📚 References

- [TTC4900 Dataset](https://www.kaggle.com/datasets/savasy/ttc4900)
- [Turkish Text Classification with Word2Vec](https://www.kaggle.com/code/alperenclk/for-beginner-nlp-and-word2vec)
- [Turkish BERT Model](https://huggingface.co/savasy/bert-turkish-text-classification)
- [BERT for Turkish Classification Notebook](https://www.kaggle.com/code/ayhanc/bert-multilingual-for-turkish-text-classification)

---

---

## 📄 Project Report

If you want, you can also look at the [project report](./cse431_project2_report.pdf) for more details about the methodology, implementation, results, and analysis.

---


