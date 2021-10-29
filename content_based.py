#importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#putting movies data on 'movies' dataframe
movies = pd.read_csv('movies_metadata.csv')

# print(movies['overview'])

tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')

#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
overview_matrix = tfidf.fit_transform(movies['overview'])
# print(overview_matrix)

similarity_matrix = linear_kernel(overview_matrix,overview_matrix)
# print(similarity_matrix)

#movies index mapping
mapping = pd.Series(movies.index,index = movies['original_title'])
# print(mapping)

def recommend_movies_based_on_title(title):
    movie_index = mapping[title]
    #get similarity values with other movies
    #similarity_score is the list of index and similarity matrix
    similarity_score = list(enumerate(similarity_matrix[movie_index]))
    #sort in descending order the similarity score of movie inputted with all the other movies
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    # Get the scores of the 15 most similar movies. Ignore the first movie.
    similarity_score = similarity_score[1:15]
    #return movie names using the mapping series
    movie_indices = [i[0] for i in similarity_score]
    return (movies['original_title'].iloc[movie_indices])

recommended = recommend_movies_based_on_title('Batman Returns')

print(recommended)