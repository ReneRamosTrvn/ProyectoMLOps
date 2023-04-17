from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# import dataset streaming
df = pd.read_csv('./datasets/streaming.csv')

# This function will map the platform to the starting letter of the show_id
# Ex. netflix -> n


def get_platform_letter(platform):
    platform_map = {'netflix': 'n', 'disney': 'd', 'amazon': 'a', 'hulu': 'h'}
    return platform_map[platform]


@app.get("/")
async def root():
    return {"message": "Welcome to my project"}


@app.get('/get_max_duration/{year}/{platform}/{duration_type}')
def get_max_duration(year: int, platform: str, duration_type: str):
    """
    Get the title of the movie which contains the longest duration in certain year and platform.

    Parameters:
    year (int): The year of release of the movie.
    platform (str): The name of the platform (netflix, disney, amazon, hulu).
    duration_type (str): The type of duration ('Min' or 'Season').

    Returns:
    A dictionary with the key 'moviea' and the value is the title of the movie.
    """
    platform_letter = get_platform_letter(platform)
    df_filtered = df.loc[(df['release_year'] == year) & df['show_id'].str.startswith(
        platform_letter) & (df['duration_type'] == duration_type) & (df['type'] == 'movie')]
    movie = df_filtered.loc[df_filtered['duration_int'].idxmax()]
    return {'movie': movie['title']}


@app.get('/get_score_count/{platform}/{score}/{year}')
def get_max_score(platform: str, score: float, year: int):
    """
    Get the quantity of movies that have a rating score more than XX in a given platform and year.

    Parameters:
    platform (str): The name of the platform (netflix, disney, amazon, hulu).
    score (float): The minimum score that the movie should have.
    year (int): The year of release of the movie.

    Returns:
    A dictionary with the keys 'plataform', 'quantity',
    'year', and 'score'. The values are the name of the platform, the quantity of
    movies that match the criteria, the year of release of the movies, and the minimum score.
    """
    platform_letter = get_platform_letter(platform)
    df_sc = df.loc[df['show_id'].str.startswith(platform_letter) & (
        df['score'] > score) & (df['release_year'] == year) & (df['type'] == 'movie')]
    return {'paltaform': platform, 'quantity': df_sc.shape[0], 'year': year, 'score': score}


@app.get('/get_count_platform/{platform}')
def get_count_platform(platform: str):
    """
    Get the quantity of movies of a given platform.

    Parameters:
    platform (str): The name of the platform (netflix, disney, amazon, hulu).

    Returns:
    A dictionary with the keys 'plataform' and 'movies'.
    The values are the name of the platform and the quantity of movies available on that platform.
    """
    platform_letter = get_platform_letter(platform)
    df_pl = df.loc[df['show_id'].str.startswith(platform_letter)]
    return {'plataform': platform, 'movies': df_pl.shape[0]}


@app.get('/get_actor/{platform}/{year}')
def get_actor(platform: str, year: int):
    """
    Get the most repeated actor of a given platform on a given year

    Parameters: 
    platform (str): The name of the platform (netflix, disney, amazon, hulu).

    Returns:
    A dictionary with the keys platform, year, actor, participations
    """
    platform_letter = get_platform_letter(platform)
    df_ac = df.loc[df['show_id'].str.startswith(
        platform_letter) & (df['release_year'] == year)]
    actors = df_ac['cast'].apply(lambda x: x.split(
        ', ') if isinstance(x, str) else [])
    flattened_actors = [
        actor for actors_list in actors for actor in actors_list]
    if flattened_actors:
        most_common_actor = max(set(flattened_actors),
                                key=flattened_actors.count)
        apariciones = flattened_actors.count(most_common_actor)
    else:
        most_common_actor = None
        apariciones = 0
    return {
        'plataform': platform,
        'year': year,
        'actor': most_common_actor,
        'participations': apariciones
    }


@app.get('/prod_per_country/{type}/{country}/{year}')
def prod_per_country(type: str, country: str, year: int):
    """
    Get the quantity of movies or TV shows of a given country on a given year

    Parameters:
    type (str): The type of the product (ex. movie, documental, tv show)
    country (str): The country you want to get the products from
    year (int): The year of release of the movie.

    Returns:
    A dictinary with the keys country, year and movies
    """
    df_pc = df.loc[(df['type'] == type) & (
        df['release_year'] == year) & (df['country'] == country)]
    return {'country': country, 'year': year, 'movies': df_pc.shape[0]}

# Get all movies of a given rating


@app.get('/get_contents/{rating}')
def get_contents(rating: str):
    """
    Get all movies of a given rating

    Parameters:
    rating (float): The rating you want to filter the data

    Returns:
    A dictionary with the keys rating and content, whith content being the
    number of products on with the same rating
    """
    df_rat = df.loc[df['rating'] == rating]
    count = df_rat['rating'].count()
    return {'rating': rating, 'content': count}


@app.get('/get_recommendations/{title}')
def get_recommendations(title: str):
    """
    This model will give you a recommendation based on a given movie

    Parameters:
    title (str): The title of the movie you want to get recommendations from

    Returns:
    A list of 5 movies this model recommends you
    """
    num_recommendations = 5
    # Load preprocessed data and trained model from joblib files
    df_sample, features, cosine_sim = joblib.load('preprocessed_data.joblib')
    scaler = joblib.load('trained_model.joblib')

    # Preprocess the input title
    title_features = pd.DataFrame(
        features.iloc[df_sample[df_sample['title'] == title].index[0]]).T
    title_features_scaled = scaler.transform(title_features)

    # Calculate cosine similarity between input title and all other titles
    sim_scores = list(
        enumerate(cosine_sim[df_sample[df_sample['title'] == title].index[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    # Get recommended movie titles
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = df_sample['title'].iloc[movie_indices].tolist()

    return recommended_movies
