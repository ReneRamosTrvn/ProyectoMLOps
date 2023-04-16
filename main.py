from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

app = FastAPI()

df = pd.read_csv('./datasets/streaming.csv')


def get_platform_letter(platform):
    platform_map = {'netflix': 'n', 'disney': 'd', 'amazon': 'a', 'hulu': 'h'}
    return platform_map[platform]


@app.get('/get_max_duration/{year}/{platform}/{duration_type}')
def get_max_duration(year: int, platform: str, duration_type: str):
    platform_letter = get_platform_letter(platform)
    df_filtered = df.loc[(df['release_year'] == year) & df['show_id'].str.startswith(
        platform_letter) & (df['duration_type'] == duration_type) & (df['type'] == 'movie')]
    movie = df_filtered.loc[df_filtered['duration_int'].idxmax()]
    return {'pelicula': movie['title'][0]}


@app.get('/get_score_count/{platform}/{score}/{year}')
def get_max_score(platform: str, score: float, year: int):
    platform_letter = get_platform_letter(platform)
    df_sc = df.loc[df['show_id'].str.startswith(platform_letter) & (
        df['score'] > score) & (df['release_year'] == year) & (df['type'] == 'movie')]
    return {'paltaforma': platform, 'cantidad': df_sc.shape[0], 'anio': year, 'score': score}


@app.get('/get_count_platform/{platform}')
def get_count_platform(platform: str):
    platform_letter = get_platform_letter(platform)
    df_pl = df.loc[df['show_id'].str.startswith(platform_letter)]
    return {'plataforma': platform, 'peliculas': df_pl.shape[0]}


@app.get('/get_actor/{platform}/{year}')
def get_actor(platform: str, year: int):
    platform_letter = get_platform_letter(platform)
    df_ac = df.loc[df['show_id'].str.startswith(
        platform_letter) & (df['release_year'] == year)]
    actors = df_ac['cast'].str.split(', ')
    most_common_actor = actors.value_counts().idxmax()[0]
    return {
        'plataforma': platform,
        'anio': year,
        'actor': most_common_actor,
        'apariciones': most_common_actor
    }


@app.get('/prod_per_country/{type}/{country}/{year}')
def prod_per_country(type: str, country: str, year: int):
    df_pc = df.loc[(df['type'] == type) & (
        df['release_year'] == year) & (df['country'] == country)]
    return {'pais': country, 'anio': year, 'peliculas': df_pc.shape[0]}


@app.get('/get_rating/{rating}')
def get_contents(rating: str):
    df_rat = df.loc[df['rating'] == rating]
    count = df_rat['rating'].count()
    return {'rating': rating, 'contenido': count}


@app.get('/get_recommendations_new/{title}')
def get_recommendations_new(title: str, num_recommendations=5):
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