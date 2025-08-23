import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(movie_title, movies_filtered, content_matrix, top_n=5):
    """
    Recommends similar movies based on content features.
    
    Parameters:
    - movie_title (str): The reference movie title.
    - movies_filtered (pd.DataFrame): DataFrame with columns ['movieId', 'title', 'genres'].
    - content_matrix (pd.DataFrame): DataFrame indexed by movieId with features + 'cluster'.
    - top_n (int): Number of recommendations.
    
    Returns:
    - pd.DataFrame: Recommended movies with ['title', 'genres'].
    """
    # Find the movie index
    if movie_title not in movies_filtered['title'].values:
        raise ValueError(f"Movie '{movie_title}' not found in dataset.")

    idx = movies_filtered[movies_filtered['title'] == movie_title].index[0]
    movie_id = movies_filtered.loc[idx, 'movieId']

    # Find cluster of the reference movie
    cluster = content_matrix.loc[movie_id, 'cluster']

    # Get movieIds in the same cluster (excluding reference movie)
    same_cluster_ids = content_matrix[content_matrix['cluster'] == cluster].index
    same_cluster_ids = same_cluster_ids[same_cluster_ids != movie_id]

    # Compute cosine similarity
    ref_vector = content_matrix.loc[movie_id].drop('cluster').values.reshape(1, -1)
    cluster_matrix = content_matrix.loc[same_cluster_ids].drop('cluster', axis=1).values

    similarities = cosine_similarity(ref_vector, cluster_matrix)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    recommended_ids = same_cluster_ids[top_indices]

    # Return recommended movies
    return movies_filtered[movies_filtered['movieId'].isin(recommended_ids)][['movieId','tmdbId', 'title', 'genres']]
