# 📰 FAKE NEWS CLASSIFIER

A **Machine Learning** project that classifies news articles as **FAKE** or **REAL** using **Natural Language Processing (NLP)** and **scikit-learn**.

Fake news has become a critical issue in the digital age. This project uses **TF-IDF vectorization** to transform text data into features and a **Linear Support Vector Classifier (LinearSVC)** to build an effective misinformation detection model.

---

## 🔍 OVERVIEW

This project demonstrates how to build a **text classification model** for detecting fake news. It follows a complete ML pipeline:

✅ Load and preprocess the dataset  
✅ Clean and normalize the text  
✅ Transform text using **TF-IDF**  
✅ Train a **LinearSVC** model  
✅ Evaluate performance (accuracy, confusion matrix, precision, recall, F1-score)  
✅ Visualize results with **WordClouds** and **Confusion Matrix heatmaps**

---

## 📊 DATASET

The dataset contains labeled news articles, with each entry classified as **FAKE** or **REAL**.

**Sample Structure:**

| title                       | text                                  | label |
|-----------------------------|----------------------------------------|--------|
| Donald Trump Sends Out...   | Donald Trump has reportedly sent...   | FAKE   |
| The economy is improving... | Reports indicate that the economy...  | REAL   |

📌 **Dataset Source**: [Fake News Dataset on Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## ✨ FEATURES

✔️ Text preprocessing (stopword removal, lemmatization, cleaning special characters)  
✔️ TF-IDF vectorization  
✔️ Linear Support Vector Machine (SVM) classifier  
✔️ Model evaluation (accuracy, precision, recall, F1-score)  
✔️ WordCloud visualizations for FAKE and REAL news  
✔️ Confusion Matrix heatmap

---

## 🔄 PROJECT WORKFLOW

### 🧹 1. DATA PREPROCESSING

- Expand contractions (e.g., *can't* → *cannot*)  
- Remove punctuation and special characters  
- Convert text to lowercase  
- Remove stopwords  
- Apply lemmatization  

### 📐 2. FEATURE EXTRACTION

- Apply `TfidfVectorizer` to convert text into numerical features  

### 🧠 3. MODEL TRAINING

- Train a `LinearSVC` classifier using the training set  

### 📊 4. MODEL EVALUATION

- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, F1-Score  

### 🎨 5. VISUALIZATION

- WordClouds for FAKE and REAL news  
- Confusion Matrix heatmap  

---

## 📂 PROJECT STRUCTURE
``` bash
.
├── fake_news_classifier.ipynb    # Jupyter Notebook with full project
├── README.md                     # Documentation

```

### 🛠️ TECH STACK

• scikit-learn  
  - LinearSVC  
  - TfidfVectorizer  
  - Evaluation metrics (accuracy_score, confusion_matrix, classification_report)  

• NumPy  

• Pandas  

• Matplotlib  

• Seaborn  

• WordCloud  

• Pillow  


---

## 🚀 FUTURE IMPROVEMENTS

🤖 Try other ML models (PassiveAggressiveClassifier, Logistic Regression, Naive Bayes)  
🧠 Use deep learning models (LSTM, BERT, DistilBERT) for improved accuracy  
🌐 Deploy as a web app using Streamlit, Flask, or Django  
🔗 Add real-time API integration to classify live news headlines  
