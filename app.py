import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
import os
from recommendations import content_based_recommendation, get_similar_users, predict_rating, collaborative_filtering_recommendations

load_dotenv()

# Load datasets
links = pd.read_csv("ml-32m/links.csv")
movies_filtered = pd.read_csv("movies_filtered.csv")

# Merge with links dataset to get TMDB IDs
movies_merged = movies_filtered.merge(links, on='movieId', how='left')

content_matrix = pd.read_csv("content_matrix.csv",index_col='movieId')

small_user_item_matrix = pd.read_csv("small_user_item_matrix.csv", index_col=0)
small_user_similarity = pd.read_csv("small_user_similarity.csv", index_col=0)

# Map user/movie IDs to indices
user_to_index = {user: idx for idx, user in enumerate(small_user_item_matrix.index)}
index_to_user = {idx: user for user, idx in user_to_index.items()}
movie_to_index = {movie: idx for idx, movie in enumerate(small_user_item_matrix.columns)}
index_to_movie = {idx: movie for movie, idx in movie_to_index.items()}

# Compute user means
user_means = small_user_item_matrix.replace(0, pd.NA).mean(axis=1).to_dict()
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
    """
)
method = st.sidebar.selectbox("Select Recommendation Method", ["Content-Based","Collaborative Filtering"])

user_id = None
movies_per_row = 3

if method in ["Content-Based"]:
    num_movies = st.sidebar.slider("Number of Movies", 2, 5, 10)
    selected_movie = st.selectbox("Type or select a movie from the dropdown menu", movies_merged['title'].values)
elif method == "Collaborative Filtering":
    user_id = st.text_input("Enter MovieLens User ID for Collaborative Filtering" if method == "Collaborative Filtering" else None)

if st.button("Recommend movies"):
    if method == "Content-Based":

        recommended_movies = content_based_recommendation(selected_movie, movies_merged, content_matrix, top_n=num_movies)
        cols_per_row = 3
        for i in range(0, len(recommended_movies), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(recommended_movies):  
                    with cols[j]:
                        st.write(f"**{recommended_movies.iloc[i + j]['title']}**")
                        movie = recommended_movies.iloc[i + j]
                        st.image(fetch_movie_poster(movie['tmdbId']))
                        st.caption(f"**{movie['title']}**\n\n{movie['genres']}")


    elif method == "Collaborative Filtering":
        if user_id:
            try:
                user_id_int = int(user_id)
                recommendations = collaborative_filtering_recommendations(
                    user_id_int, movies_filtered,
                    user_to_index, small_user_item_matrix,
                    user_means, movie_to_index,
                    small_user_similarity,
                    get_similar_users,
                    n=5, k=10
                )

                st.subheader("Recommended Movies")
                cols_per_row = movies_per_row
                for i in range(0, len(recommendations), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(recommendations):
                            movie = recommendations.iloc[i+j]
                            with cols[j]:
                                st.write(f"**{movie['title']}**")
                                poster = fetch_movie_poster(movie.get('tmdbId', None))
                                if poster:
                                    st.image(poster)
                                st.caption(movie['genres'])

                # Show movies the user liked before
                liked_movies = small_user_item_matrix.loc[user_id_int]
                liked_movies = liked_movies[liked_movies > 0].index
                liked_df = movies_filtered[movies_filtered['movieId'].isin(liked_movies)]
                st.subheader("Movies you liked before")
                st.write(liked_df[['title','genres']])
            except Exception as e:
                st.error(f"Error: {e}")