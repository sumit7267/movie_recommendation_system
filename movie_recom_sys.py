import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle



credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

credits.head()
movies.head(1)

movies = movies.merge(credits, left_on='title', right_on='title')

movies = movies[['movie_id' , 'title' , 'overview' , 'genres' , 'keywords' , 'cast' , 'crew']]

movies.head(1)

def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres'] = movies['genres'].apply(convert)

# print(movies['genres'])

movies['keywords'] = movies['keywords'].apply(convert)

# print(movies['keywords'])

movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]]) #for top 3 actors of a movie

movies['crew']=movies['crew'].apply(lambda x:[i['name'] for i in ast.literal_eval(x) if i['job']=='Director']) # only taking director from the crew

movies['tag'] = movies['genres'] + movies['cast'] + movies['crew'] + movies['keywords']

movies =  movies[['movie_id' , 'title' , 'overview' , 'tag']]

movies['tag'] = movies['tag'].apply(lambda x: " ".join(x))
movies['tag'] = movies['tag'].apply(lambda x: x.lower())

# print(movies['tag'])
# print(movies.head())

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tag'])

cosin_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendation(title, cosin_sim=cosin_sim):
  idx = movies[movies['title'] == title].index[0]
  sim_scores = list(enumerate(cosin_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1] , reverse=True)
  sim_scores = sim_scores[1:11] # top 10 similar movies
  movie_indices = [i[0] for i in sim_scores]
  return movies['title'].iloc[movie_indices]



# print(get_recommendation('The Dark Knight Rises'))

with open('movie_data.pkl', 'wb') as file:
  pickle.dump((movies, cosin_sim), file)