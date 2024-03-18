from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

from utils.preprocess import clean_similar_texts, finalpreprocess, generate_oversampled_data

def get_treated_data():
    df1 = pd.read_csv('spam.csv')
    df1['Category'] = df1['Category'].map({'ham': 0, 'spam': 1})
    df1 = df1.dropna(subset=['Message'])

    df2 = pd.read_csv('spam2.csv', usecols=['Body', 'Label'])
    df2 = df2.rename(columns={'Body': 'Message', 'Label': 'Category'})
    df2 = df2.dropna(subset=['Message'])
    
    df = pd.concat([df1, df2], ignore_index=True)
    df = clean_similar_texts(df)
    df = generate_oversampled_data(df)

    df['Message'] = df['Message'].apply(lambda x: finalpreprocess(x))

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Message'])
    y = df['Category']

    return train_test_split(X, y, test_size=0.2, random_state=42)
