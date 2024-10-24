# 📊 Twitter Sentiment Analysis (😀 Positive Tweets - 😞 Negative Tweets)

## 📄 Project Description
This project performs sentiment analysis on 🐦 tweets, classifying them as either 😀 positive or 😞 negative. The **Sentiment140 dataset** from Kaggle is used.

## 📂 Dataset Link
You can download the dataset from Kaggle:  
[📊 Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

### 📋 Dataset Overview
- **🎯 target**: Sentiment of the tweet (`0` = 😞 Negative, `1` = 😀 Positive)
- **🆔 id**: Unique ID of the tweet
- **📅 date**: Date and time when the tweet was posted
- **🚩 flag**: Not used in the analysis
- **👤 user**: Username of the person who posted the tweet
- **💬 text**: The tweet itself

---

## 🌟 Project Overview
The tweets are cleaned 🧽, tokenized ✂️, stemmed 🌱, and converted into numerical features 🔢 using the **TF-IDF vectorizer**. A **Logistic Regression** model is trained 🏋️‍♂️ on this data to classify the sentiment of the tweets.

The model achieves **77.8% accuracy** 📊 on unseen test data.

---

## 📦 Requirements
To run this project, you need the following Python 🐍 libraries installed:

```bash
pip install pandas numpy matplotlib nltk scikit-learn
```

Additionally, download the **NLTK stopwords** by running this command:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## 📝 Code Walkthrough

### 1. 📚 Import Libraries
The necessary libraries for data loading 📥, cleaning 🧽, processing 🔄, and model building 🏗️ are imported:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
```

### 2. 📥 Load the Dataset
Load the dataset into a pandas DataFrame. Ensure proper encoding for text data:

```python
twitter_data = pd.read_csv('path_to_dataset.csv', encoding='ISO-8859-1')
```

### 3. 🧽 Preprocessing the Data
Perform the following steps:
- **📝 Rename Columns**: Define proper column names as the dataset does not come with headers.
- **🔄 Replace target labels**: The original dataset uses `4` for 😀 positive sentiment, which is converted to `1`.
- **⚠️ Handle missing values**: Identify and handle any missing data appropriately.

```python
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('path_to_dataset.csv', names=column_names, encoding='ISO-8859-1')
twitter_data.replace({'target': {4: 1}}, inplace=True)
```

### 4. 🌱 Stemming Function
Define a function to clean the text data by removing non-alphabetic characters 🔤, converting to lowercase 🔡, removing stopwords ❌, and applying stemming using the **PorterStemmer**.

```python
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)
```

Apply this function to the text column:

```python
twitter_data['text'] = twitter_data['text'].apply(stemming)
```

### 5. 🔢 Convert Text Data to Numerical Data
Convert the cleaned text data into numerical features using **TF-IDF Vectorizer**:

```python
vectorizer = TfidfVectorizer()
X = twitter_data['text'].values
Y = twitter_data['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
```

### 6. 🏋️‍♂️ Train the Logistic Regression Model
Train a **Logistic Regression** model on the transformed data:

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

### 7. 📊 Evaluate the Model
Evaluate the model’s accuracy on the test set:

```python
X_train_prediction = model.predict(X_test)
training_data_accuracy = accuracy_score(Y_test, X_train_prediction)
print('Accuracy score on the training data = ', training_data_accuracy)
```

**Model Accuracy**: 77.8% 📈

### 8. 💾 Save the Model
Save the trained model using the **pickle** module for future use:

```python
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
```

---

## 📊 Summary of Results
- **Logistic Regression Model**: Achieves an accuracy of **77.8%** 📈 on the test set.
- **TF-IDF Vectorization**: Converts text data into a numerical format suitable for model training.
- **💾 Model Persistence**: The trained model is saved using `pickle` for reuse without retraining.

---

## 🚀 How to Run the Project
1. 🔄 Clone the repository or download the files.
2. 📦 Install the required libraries listed in the **Requirements** section.
3. 📥 Download the dataset from Kaggle and load it into the project.
4. ▶️ Run the Python script to preprocess the data, train the model, and evaluate the results.
5. 💾 The trained model will be saved as `trained_model.sav` for future use.

