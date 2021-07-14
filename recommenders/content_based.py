"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv')
ratings = pd.read_csv('resources/data/ratings.csv')
imdb = pd.read_csv('resources/data/imdb_data.csv')
tags = pd.read_csv('resources/data/tags.csv')



movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """

    #convert all the tags to strings and lower case them
    tags['tag'] = tags['tag'].map(str).str.lower()
    
    #grouping tags based on movieId
    grouped_tags = tags.groupby('movieId')['tag'].apply(' '.join).reset_index()

    movies_imdb = pd.merge(movies,imdb, on='movieId',how='left')

    movies_imdb_tags = pd.merge(movies_imdb,grouped_tags, on='movieId',how='left')
    
    
    """This part of the function is for cleaning genres column
    """
    #replace separators with space
    pattern = r'\|'
    movies_imdb_tags['genres'] = movies_imdb_tags['genres'].apply(lambda x: re.sub(pattern, ' ', str(x)))
    
    #lowercase
    movies_imdb_tags['genres'] = movies_imdb_tags['genres'].str.lower()
    
    
    """This cleans and retrieves and the lead actor/actress from the title_cast 
    """
    #converting to string
    movies_imdb_tags['title_cast'] = movies_imdb_tags['title_cast'].apply(lambda x : str(x))
    
    #creating list of cast members based on separator
    movies_imdb_tags['title_cast'] = movies_imdb_tags['title_cast'].str.split('|')
    
    #taking first item of list
    movies_imdb_tags['title_cast'] =  movies_imdb_tags['title_cast'].apply(lambda x: x[0:5])

    movies_imdb_tags['title_cast'] =  movies_imdb_tags['title_cast'].apply(lambda x: " ".join(x))
    #removing space
    pattern = r'\s{1,}'
    movies_imdb_tags['title_cast'] = movies_imdb_tags['title_cast'].apply(lambda x: re.sub(pattern, '', str(x)))
    
    #lowercasing
    movies_imdb_tags['title_cast'] = movies_imdb_tags['title_cast'].str.lower()

    #renaming column
    movies_imdb_tags.rename(columns ={'title_cast':'5lead_actors'},inplace = True)
    
    """This part of the function cleans the directors column
    """
    #removing spaces
    pattern = r'\s{1,}'
    movies_imdb_tags['director'] = movies_imdb_tags['director'].apply(lambda x: re.sub(pattern, '', str(x)))
    
    # removing commas and dashes
    pattern = r'\.|\-'
    movies_imdb_tags['director'] = movies_imdb_tags['director'].apply(lambda x: re.sub(pattern, '', str(x)))

    #lowercasing
    movies_imdb_tags['director'] = movies_imdb_tags['director'].str.lower()
    
    """This part of the function cleans the plot_keywords
    """
    #replacing separator with spaces
    pattern = r'\|'
    movies_imdb_tags['plot_keywords'] = movies_imdb_tags['plot_keywords'].apply(lambda x: re.sub(pattern, ' ', str(x)))

    #creating year column
    pattern = r'\(+\d+\)'
    movies_imdb_tags['year'] = movies_imdb_tags['title'].apply(lambda x: ''.join(re.findall(pattern, x)))

    pattern = r'\(|\)'
    movies_imdb_tags['year'] = movies_imdb_tags['year'].apply(lambda x: re.sub(pattern,'',x))


    movies_imdb_tags['documents'] = movies_imdb_tags['genres']+" "+movies_imdb_tags['5lead_actors']+" "+movies_imdb_tags['director']+" "+movies_imdb_tags['plot_keywords']+" "+movies_imdb_tags['tag']+" "+movies_imdb_tags['year']

    movies_imdb_tags['documents'] =  movies_imdb_tags['documents'].map(str)

    # Subset of the data
    movies_subset = movies_imdb_tags[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000) #27000
    # Instantiating and generating the count matrix
    count_vec =  TfidfVectorizer(stop_words=['nan','Nan','NAN','NaN','np.nan'],analyzer='word')
    count_matrix = count_vec.fit_transform(data['documents'].apply(lambda x: np.str_(x)))
    indices = pd.Series(data['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)

    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies
