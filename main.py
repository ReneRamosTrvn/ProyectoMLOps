# from typing import Union

from fastapi import FastAPI
import pandas as pd

app = FastAPI()

df = pd.read_csv('./datasets/streaming.csv')


@app.get('/get_max_duration/{year}/{platform}/{duration_type}')
def get_max_duration(year: int, platform: str, duration_type: str):
    df_filtered = df.loc[(df['release_year'] == year) & df['show_id'].str.startswith(
        platform) & (df['duration_type'] == duration_type) & (df['type'] == 'movie')]
    movie = df_filtered.loc[df_filtered['duration_int'].idxmax()]
    return movie['title']


@app.get('/get_score_count/{platform}/{score}/{year}')
def get_max_score(platform: str, score: float, year: int):
    df_sc = df.loc[df['show_id'].str.startswith(platform) & (
        df['score'] > score) & (df['release_year'] == year) & (df['type'] == 'movie')]
    return df_sc.shape[0]


@app.get('/get_count_platform/{platform}')
def get_count_platform(platform: str):
    df_pl = df.loc[df['show_id'].str.startswith(platform)]
    return df_pl.shape[0]


@app.get('/get_actor/{platform}/{year}')
def get_actor(platform: str, year: int):
    df_ac = df.loc[df['show_id'].str.startswith(
        platform) & (df['release_year'] == year)]
    actors = df_ac['cast'].str.split(', ')
    most_common_actor = actors.value_counts().idxmax()[0]
    return most_common_actor


@app.get('/prod_per_country/{type}/{country}/{year}')
def prod_per_country(type: str, country: str, year: int):
    df_pc = df.loc[(df['type'] == type) & (
        df['release_year'] == year) & (df['country'] == country)]
    dicc = {
        'country': country,
        'type': type,
        'products': df_pc.shape[0]
    }
    return dicc


@app.get('/get_rating/{rating}')
def get_contents(rating: str):
    df_rat = df.loc[df['rating'] == rating]
    count = df_rat['rating'].count()
    return count
