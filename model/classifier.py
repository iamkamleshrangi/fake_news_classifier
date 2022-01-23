import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from sklearn.metrics import confusion_matrix
import numpy as np
from joblib import dump, load

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

true_df = true_df[['text']]
true_df['type'] = 1

fake_df = fake_df[['text']]
fake_df['type'] = 0

df = pd.concat([true_df, fake_df])

def get_clean(text):
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]',' ', text)
    text = re.sub('\n+|\s+',' ', text)
    text = word_tokenize(text)
    text = [ i for i in text if i not in stopwords.words('english') ]
    text = [ lemmatizer.lemmatize(i) for i in text ]
    print('R -> ',text)
    return text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

X = df['text']
y = df['type']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.30)
vect = CountVectorizer(tokenizer=get_clean)
tfidf = TfidfTransformer()
clf = RandomForestClassifier()

X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
clf.fit(X_train_tfidf, y_train)

# predict on test data
X_test_counts = vect.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)

labels = np.unique(y_pred)
confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
accuracy = (y_pred == y_test).mean()

print("Labels:", labels)
print("Confusion Matrix:\n", confusion_mat)
print("Accuracy:", accuracy)

dump(clf, 'classifier.joblib')

