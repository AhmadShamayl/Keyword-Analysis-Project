Python Keyword Analysis Project
This Python project utilizes various natural language processing tools to perform keyword analysis, clustering, and phrase extraction from a dataset. It leverages TfidfVectorizer from Scikit-learn, DBSCAN for clustering, and Word2Vec and FastText from Gensim for semantic analysis.

Features:
Keyword Clustering: Clusters keywords using TF-IDF vectors and DBSCAN.
Top Terms Extraction: Extracts top terms from each cluster.
Semantic Analysis: Uses Word2Vec and FastText models to find semantically similar words.
N-gram Analysis: Extracts long-tail keywords using CountVectorizer.

Requirements:
Python 3.8+
pandas
scikit-learn
gensim
numpy

Installation:
git clone https://github.com/AhmadShamayl/Keyword-Analysis-Project.git

Navigate to the project directory:
cd keyword-analysis

Install the required packages:
pip install -r requirements.txt

Usage:
Prepare your dataset in an Excel file with a column named 'Keyword'.
Place the Excel file in the project directory.

Run the script:
python keyword_analysis.py

Example
Assuming you have a dataset with keywords, the script performs the following actions:

Load Data: Keywords are loaded from 'Keywords_check_GT.xlsx'.
TF-IDF Transformation: Keywords are transformed into TF-IDF vectors.
Clustering: Keywords are clustered using the DBSCAN algorithm based on cosine similarity of TF-IDF vectors.
Semantic Analysis: Finds similar words to "green top" using FastText.
Long Tail Keyword Extraction: Extracts top long-tail keywords based on n-gram frequency.
Output

The script outputs:
A CSV file all_clusters.csv with keywords and their corresponding cluster IDs.
Prints top terms for each cluster in the console.
Displays most similar words to "green top".
Shows top long-tail keywords.
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or improvements.

