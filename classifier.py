# ============================================
# PHISHING EMAIL CLASSIFIER
# ============================================

# --- 1. Import Libraries ---
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 2. Load Dataset ---
df = pd.read_csv("data/CEAS_08.csv")

# --- 3. Preprocessing ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['subject'] = df['subject'].fillna('')
df['text'] = df['subject'] + ' ' + df['body']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# --- 4. Feature Engineering ---
X = df['clean_text']
y = df['label']
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# --- 5. Train Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42)

# --- 6. Model Training ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 7. Evaluation ---
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['Legit', 'Phishing']))
