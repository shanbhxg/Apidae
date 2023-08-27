import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('job_data.csv') 
data.dropna(subset=['text'], inplace=True)

job_descriptions = data['text']
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(job_descriptions)
print(tfidf_matrix) # sparse
dense_tfidf_array = tfidf_matrix.toarray() # dense