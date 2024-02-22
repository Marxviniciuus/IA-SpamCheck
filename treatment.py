from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.text import clean_text


def getTreatedData():
    df = pd.read_csv('spam.csv')
    df['Message'] = df['Message'].apply(clean_text)
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Message'])
    y = df['Category']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
