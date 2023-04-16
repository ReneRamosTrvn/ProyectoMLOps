MLOps Individual Project

Overview
This is a streaming platform project that allows you to search and explore movies across different streaming platforms. The project uses machine learning algorithm to recommend movies based on user preferences and provides users with useful information:

You can access the API here: https://proyecto-ml-ops.onrender.com

You have multiple functions you can get data from:

/get_max_duration/{year}/{platform}/{duration_type}
Get the title of the movie which contains the longest duration in certain year and platform.

    Parameters:
    year (int): The year of release of the movie.
    platform (str): The name of the platform (netflix, disney, amazon, hulu).
    duration_type (str): The type of duration ('Min' or 'Season').

    Returns:
    A dictionary with the key 'moviea' and the value is the title of the movie.
    
 /get_score_count/{platform}/{score}/{year}
 Get the quantity of movies that have a rating score more than XX in a given platform and year.

    Parameters:
    platform (str): The name of the platform (netflix, disney, amazon, hulu).
    score (float): The minimum score that the movie should have.
    year (int): The year of release of the movie.

    Returns:
    A dictionary with the keys 'plataform', 'quantity',
    'year', and 'score'. The values are the name of the platform, the quantity of
    movies that match the criteria, the year of release of the movies, and the minimum score.
    
  /get_count_platform/{platform}
  Get the quantity of movies of a given platform.

    Parameters:
    platform (str): The name of the platform (netflix, disney, amazon, hulu).

    Returns:
    A dictionary with the keys 'plataform' and 'movies'.
    The values are the name of the platform and the quantity of movies available on that platform.
    
  /get_actor/{platform}/{year}
  Get the most repeated actor of a given platform on a given year

    Parameters: 
    platform (str): The name of the platform (netflix, disney, amazon, hulu).

    Returns:
    A dictionary with the keys platform, year, actor, participations
    
  /prod_per_country/{type}/{country}/{year}
  Get the quantity of movies or TV shows of a given country on a given year

    Parameters:
    type (str): The type of the product (ex. movie, documental, tv show)
    country (str): The country you want to get the products from
    year (int): The year of release of the movie.

    Returns:
    A dictinary with the keys country, year and movies
    
  /get_contents/{rating}
  Get all movies of a given rating

    Parameters:
    rating (float): The rating you want to filter the data

    Returns:
    A dictionary with the keys rating and content, whith content being the
    number of products on with the same rating
    
  /get_recommendations_new/{title}
  This model will give you a recommendation based on a given movie

    Parameters:
    title (str): The title of the movie you want to get recommendations from

    Returns:
    A list of 5 movies this model recommends you
    
  Try: /get_contents/joni
  
  
 Technologies Used
  Python
    Numpy
    Pandas
    Scikit Learn
    Pickle
  FastAPI
  Machine Learning
