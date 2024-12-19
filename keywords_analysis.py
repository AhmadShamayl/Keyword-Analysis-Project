import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load the data
df = pd.read_excel('Keywords_check_GT.xlsx')
content = df['Keyword'].fillna('')
tfidf = TfidfVectorizer(max_features=100, stop_words='english' , ngram_range=(1,3))
tfidf_matrix = tfidf.fit_transform(content)
feature_names = tfidf.get_feature_names_out()

batch_size = 100  
n_batches = len(content) // batch_size + (len(content) % batch_size != 0)

def process_batch(batch_matrix, batch_df, batch_index):
    X = batch_matrix.toarray()  
    cosine_sim = cosine_similarity(X)
    distance_matrix = 1 - cosine_sim
    distance_matrix = np.clip(distance_matrix, 0, None)  

    dbscan = DBSCAN(eps=0.3, min_samples=3, metric='precomputed')
    clusters = dbscan.fit_predict(distance_matrix)
    batch_df['Cluster'] = clusters

    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            print(f"Batch {batch_index}: Noise/Outliers terms")
            continue
        cluster_content = batch_df[batch_df['Cluster'] == cluster_id]['Keyword']
        cluster_vector = tfidf.transform(cluster_content)
        top_terms = [feature_names[i] for i in cluster_vector.sum(axis=0).A1.argsort()[-10:]]
        print(f"Batch {batch_index}, Cluster {cluster_id} top terms: {top_terms}")

for i in range(n_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(content))
    batch_matrix = tfidf_matrix[start:end]
    batch_df = df.iloc[start:end].copy()
    process_batch(batch_matrix, batch_df, i)

all_clusters = pd.concat([batch_df for batch_df in locals().values() if isinstance(batch_df, pd.DataFrame)])
all_clusters.to_csv('all_clusters.csv')


sentences = [doc.split() for doc in df['Keyword'].fillna("")]
w2v_model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)
print('\n\nMost similar words to "green top":')
print(w2v_model.wv.most_similar('greentop'))

vectorizer = CountVectorizer(ngram_range=(2,5) , stop_words='english')
Xx = vectorizer.fit_transform(df['Keyword'].fillna(''))
top_ngrams = Xx.sum(axis =0).A1
terms = vectorizer.get_feature_names_out()
long_tail_keywords = [(terms[i] , top_ngrams[i])for i in top_ngrams.argsort()[-10:]]
print (f'\n\nTop log tail keywords: {long_tail_keywords}')  