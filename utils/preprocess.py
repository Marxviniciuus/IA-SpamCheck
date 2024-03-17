import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import jaccard_distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

wl = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)

    return text


def jaccard_similarity(text1, text2):
    tokens1 = set(word_tokenize(text1.lower()))
    tokens2 = set(word_tokenize(text2.lower()))
    jaccard_sim = 1 - jaccard_distance(tokens1, tokens2)
    return jaccard_sim


def clean_similar_texts(df):
    similarity_threshold = 0.8
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Message'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    index_to_remove = set()

    for i in range(len(cosine_sim)):
        for j in range(i+1, len(cosine_sim)):
            if cosine_sim[i][j] > similarity_threshold:
                index_to_remove.add(j)

    df_cleaned = df.drop(index_to_remove)

    return df_cleaned


def stopword(text: str) -> str:
    a = [i for i in text.split() if i not in stopwords.words('english')]
    return ' '.join(a)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ

    if tag.startswith('V'):
        return wordnet.VERB

    if tag.startswith('N'):
        return wordnet.NOUN

    if tag.startswith('R'):
        return wordnet.ADV

    return wordnet.NOUN


def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
    return " ".join(a)


def finalpreprocess(text: str) -> str:
    return lemmatizer(stopword(preprocess(text)))
