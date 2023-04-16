from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('./datasets/streaming.csv')


def get_recommendations_new(title, df=df, num_recommendations=5):
    # Step 1: Create dummy variables for each genre
    genre_dummies = df['listed_in'].str.join('|').str.get_dummies()

    # Step 2: Concatenate with the score column
    features = pd.concat([genre_dummies, df['score']], axis=1)

    # Step 3: Calculate cosine similarity
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    cosine_sim = cosine_similarity(features_scaled)

    # Step 4: Get top recommendations
    indices = pd.Series(df.index, index=df['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
