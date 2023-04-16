import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import gzip

df = pd.read_csv('./datasets/streaming.csv')
df_sample = df.sample(n=3000, random_state=42)

df = df_sample.drop(['type', 'Unnamed: 0', 'show_id', 'director', 'country', 'date_added',
                     'release_year', 'duration_int', 'duration_type', 'description', 'cast'], axis=1)
genre_dummies = df_sample['listed_in'].str.join('|').str.get_dummies()
features = pd.concat([genre_dummies, df['score']], axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
cosine_sim = cosine_similarity(features_scaled)

with gzip.open('preprocessed_data.pickle.gz', 'wb') as f:
    pickle.dump((df_sample, features, cosine_sim), f)

with gzip.open('trained_model.pickle.gz', 'wb') as f:
    pickle.dump((scaler,), f)


"""
Model Name: Movie Similarity Recommender

Description: 
This model uses a dataset of movies and TV shows available on various streaming platforms, and creates a similarity matrix between them based on their genres and scores. It then recommends movies similar to a given movie or TV show.

Data:
The model uses a dataset of 3000 randomly sampled movies and TV shows from the original dataset. The original dataset contains information about movies and TV shows available on various streaming platforms, including Netflix, Hulu, and Prime Video.

Features:
The model considers the following features for each movie or TV show:
- Genres (encoded as one-hot vectors)
- Score (a numeric rating given to each movie or TV show)

Preprocessing:
The following preprocessing steps are performed on the dataset:
- Rows with missing data are removed
- Irrelevant columns are dropped
- Genres are one-hot encoded
- Scores are standardized using a StandardScaler

Model Training:
The model uses cosine similarity to compute the similarity between movies and TV shows based on their features. A similarity matrix is created using the cosine similarity measure.

Model Persistence:
The preprocessed data and the trained model are saved as compressed pickle files:
- preprocessed_data.pickle.gz: Contains the preprocessed dataset, features, and similarity matrix
- trained_model.pickle.gz: Contains the trained StandardScaler object used to standardize the features

Usage:
The model can be used to recommend movies or TV shows similar to a given movie or TV show. To use the model, the following steps should be performed:
- Load the preprocessed data and the trained model from the pickle files
- Retrieve the features for the given movie or TV show
- Standardize the features using the trained StandardScaler object
- Compute the cosine similarity between the standardized features and the features of all other movies or TV shows in the dataset
- Retrieve the most similar movies or TV shows based on the computed cosine similarity scores
"""
