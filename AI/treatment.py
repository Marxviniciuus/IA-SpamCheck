from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

from utils.preprocess import finalpreprocess

def get_treated_data():
    df = pd.read_csv('spam.csv')
    df['Message'] = df['Message'].apply(lambda x: finalpreprocess(x))
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Message'])
    y = df['Category']

    return train_test_split(X, y, test_size=0.2, random_state=42)
