# Cosine Similarity Calculation

This Python script demonstrates the calculation of cosine similarity between two sentences using both CountVectorizer and TfidfVectorizer from the scikit-learn library. Additionally, it applies the same technique to a dataset of movie overviews to find similarity scores between movie titles.

## Dependencies

- [scikit-learn](https://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/)

Make sure to install these dependencies using the following:

```bash
pip install scikit-learn pandas
```

## Code Explanation

### Sentence Similarity

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample sentences
sentence1 = "they placed an order for 500 airbus aircraft."
sentence2 = "they will place an order for 500 boeing aircraft."

# Vectorize sentences
vectorizer = CountVectorizer().fit_transform([sentence1, sentence2])
vectors = vectorizer.toarray()

# Calculate cosine similarity
cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0, 0]

print("Cosine Similarity: ", cosine_sim)
```

This section calculates the cosine similarity between two sample sentences.

### Movie Similarity

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load movie data
data = pd.read_csv("/Users/sevketugurel/Desktop/CosineSimilarity/IMDB.csv")
df = data.copy()
df = df.dropna(subset=["overview"])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["overview"])

# Calculate cosine similarity for all movie pairs
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
titles = df["title"].tolist()
long_data = []

# Create a DataFrame with movie pairs and their similarity scores
for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        title1 = titles[i]
        title2 = titles[j]
        score = cosine_sim[i, j]
        long_data.append([title1, title2, score])

long_df = pd.DataFrame(long_data, columns=["title1", "title2", "score"])
long_df = long_df[long_df["title1"] != long_df["title2"]]
long_df = long_df.sort_values(by="score", ascending=False)
long_df = pd.DataFrame(long_df)
print(long_df.head(10))
```

This part uses the TF-IDF vectorization to calculate the cosine similarity between movie overviews.

### Movie Similarity for a Specific Movie

```python
# Select a specific movie and find similar movies
my_list = long_df[long_df['title1'] == 'Frozen II']
print(my_list.head(10))
```

This section demonstrates how to find similar movies to a specific movie ('Frozen II' in this case).

### Movie Overviews

```python
# Display overviews for selected movies
overview = df[df["title"] == "Frozen II"]["overview"].values[0]
print("Frozen II Overview: ", overview)

overview2 = df[df["title"] == "Olaf's Frozen Adventure"]["overview"].values[0]
print("Olaf's Frozen Adventure Overview: ", overview2)
```

This part prints the overviews for the selected movies ('Frozen II' and 'Olaf's Frozen Adventure').

