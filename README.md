# 📰 FAKE NEWS CLASSIFIER

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

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

| Title                       | Text                                  | Label |
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

```
fake-news-classifier/
│
├── fake_news_classifier.ipynb    # Jupyter Notebook with full project
├── data/                         # Dataset directory
│   ├── Fake.csv
│   └── True.csv
├── models/                       # Saved models
│   └── fake_news_model.pkl
├── visualizations/               # Generated plots
│   ├── wordcloud_fake.png
│   ├── wordcloud_real.png
│   └── confusion_matrix.png
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 🛠️ TECH STACK

### **Machine Learning & NLP**
- **scikit-learn**  
  - LinearSVC  
  - TfidfVectorizer  
  - Evaluation metrics (accuracy_score, confusion_matrix, classification_report)  

### **Data Processing**
- **NumPy**  
- **Pandas**  

### **Visualization**
- **Matplotlib**  
- **Seaborn**  
- **WordCloud**  
- **Pillow**  

---

## ⚙️ INSTALLATION & SETUP

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/fake-news-classifier.git
cd fake-news-classifier
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn wordcloud pillow jupyter
```

### Step 4: Download Dataset
- Visit [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Download the dataset
- Place `Fake.csv` and `True.csv` in the `data/` directory

### Step 5: Run the Notebook
```bash
jupyter notebook fake_news_classifier.ipynb
```

---

## 🚀 USAGE

### Training the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load data
fake = pd.read_csv('data/Fake.csv')
real = pd.read_csv('data/True.csv')

# Preprocessing and training
# ... (see notebook for full code)
```

### Making Predictions

```python
# Load trained model
import pickle
with open('models/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new text
news_text = "Your news article here..."
prediction = model.predict([news_text])
print(f"Prediction: {prediction[0]}")  # Output: FAKE or REAL
```

---

## 📊 PERFORMANCE METRICS

The model achieves impressive results:

- **Accuracy**: ~92-96%
- **Precision**: ~90-95%
- **Recall**: ~90-95%
- **F1-Score**: ~91-95%

### Confusion Matrix (Example)

```
                Predicted FAKE    Predicted REAL
Actual FAKE         950                50
Actual REAL          30               970
```

---

## 📈 VISUALIZATIONS

### WordCloud - FAKE News
![WordCloud Fake](visualizations/wordcloud_fake.png)

### WordCloud - REAL News
![WordCloud Real](visualizations/wordcloud_real.png)

### Confusion Matrix
![Confusion Matrix](visualizations/confusion_matrix.png)

---

## 🚀 FUTURE IMPROVEMENTS

🤖 **Try other ML models**
- PassiveAggressiveClassifier
- Logistic Regression
- Naive Bayes
- Random Forest

🧠 **Use deep learning models**
- LSTM (Long Short-Term Memory)
- BERT (Bidirectional Encoder Representations from Transformers)
- DistilBERT
- RoBERTa

🌐 **Deploy as a web app**
- Streamlit dashboard
- Flask REST API
- Django full-stack application
- FastAPI for high performance

🔗 **Add real-time features**
- Live news headline classification
- Browser extension integration
- API integration with news sources
- Mobile application

📊 **Enhanced analytics**
- Confidence scores
- Explainable AI (LIME/SHAP)
- Multi-language support
- Fact-checking integration

---

## 🤝 CONTRIBUTING

Contributions are **welcome** and greatly appreciated! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas to Contribute
- Improve model accuracy
- Add new visualization features
- Enhance documentation
- Fix bugs and issues
- Add unit tests
- Implement new features

---

## 📝 LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 AUTHOR

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 ACKNOWLEDGMENTS

- Dataset from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Inspired by the need to combat misinformation
- Built with ❤️ using scikit-learn and Python

---

## 📚 REFERENCES

- [scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Linear SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [Fake News Research Papers](https://scholar.google.com/scholar?q=fake+news+detection)

---

## ⭐ STAR THIS REPO

If you found this project helpful, please consider giving it a ⭐️!

---

<div align="center">

**Made with 🔍 and 🤖 to fight misinformation**

[Report Bug](https://github.com/yourusername/fake-news-classifier/issues) · [Request Feature](https://github.com/yourusername/fake-news-classifier/issues)

</div>
