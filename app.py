import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from recommendations import content_based_recommendation

load_dotenv()

# Load datasets
links = pd.read_csv("ml-32m/links.csv")
movies_filtered = pd.read_csv("movies_filtered.csv")

movies = movies_filtered.merge(links, on='movieId', how='left')

content_matrix = pd.read_csv("content_matrix.csv",)

def fetch_movie_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US&api_key={os.getenv('TMDB_API_KEY')}"

    headers = {
            "accept": "application/json",
        }

    response = requests.get(url, headers=headers)

    data = response.json()
    print(data)
    return 'https://image.tmdb.org/t/p/w500'+ data['poster_path']


st.title("🎬 Movie Recommendation System")
st.sidebar.info(
    """
    This app helps you discover new movies based on your preferences.  
    - Uses collaborative filtering and/or content-based filtering.  
    - Fetches posters and details using **TMDB API**.  
    - Simple and interactive interface with Streamlit.  
    🔎 Try searching for a movie and get personalized recommendations!  
    ---
    *Built with ❤️ using Python & Streamlit.*
    """
)
method = st.sidebar.selectbox("Select Recommendation Method", ["Collaborative Filtering","Content-Based","SVD"])

user_id = None
if method in ["Collaborative Filtering"]:
    user_id = st.sidebar.text_input("Enter your MovieLens user_id", value="")
num_movies = st.sidebar.slider("Number of Movies", 2, 5, 10)
movies_per_row = 3
selected_movie = st.selectbox("Type or select a movie from the dropdown menu", movies['title'].values)


if st.button("Recommend movies"):
    if method == "Content-Based":

        recommended_movies = content_based_recommendation(selected_movie, movies, content_matrix, top_n=num_movies)
        print(recommended_movies)
        cols_per_row = 3
        for i in range(0, len(recommended_movies), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(recommended_movies):  
                    with cols[j]:
                        movie = recommended_movies.iloc[i + j]
                        st.image(fetch_movie_poster(movie['tmdbId']))
                        st.caption(f"**{movie['title']}**\n\n{movie['genres']}")