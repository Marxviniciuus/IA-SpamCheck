from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from utils.text import clean_text

df = pd.read_csv('spam.csv')
df['Message'] = df['Message'].apply(clean_text)
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})


def getTreatedData():
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Message'])
    y = df['Category']
    
    return X, y
