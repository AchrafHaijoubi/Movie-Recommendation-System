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
def get_similar_users(user_id, user_to_index, user_similarity_df, n=10):
    """
    Get top N similar users for a given user.
    """
    if user_id not in user_to_index:
        return pd.Series(dtype=float)  # empty series

    user_idx = user_to_index[user_id]
    # Get similarities from the dataframe row
    similar_users = user_similarity_df.iloc[user_idx].copy()
    
    # Exclude the user itself
    if user_id in similar_users.index:
        similar_users = similar_users.drop(labels=user_id)

    # Keep only positive similarities
    similar_users = similar_users[similar_users > 0]

    return similar_users.sort_values(ascending=False).head(n)


def predict_rating(user_id, user_to_index, user_means, user_item_matrix,
                   get_similar_users_func, movie_to_index, movie_id, user_similarity_df, k=10):
    """
    Predict rating for a user-movie pair using user-based collaborative filtering.
    """
    if user_id not in user_to_index or movie_id not in movie_to_index:
        return user_means.get(user_id, 2.5)  # default rating if user/movie not found

    user_idx = user_to_index[user_id]
    movie_idx = movie_to_index[movie_id]

    # Get similar users
    similar_users = get_similar_users_func(user_id, user_to_index, user_similarity_df, n=k*2)

    if similar_users.empty:
        return user_means[user_id]

    # Users who rated the movie
    movie_ratings = user_item_matrix.iloc[:, movie_idx]
    rated_users = movie_ratings[movie_ratings > 0].index

    # Filter similar users who rated the movie
    valid_similar_users = similar_users.loc[similar_users.index.intersection(rated_users)]

    if valid_similar_users.empty:
        return user_means[user_id]

    # Take top k similar users
    top_similar_users = valid_similar_users.head(k)

    numerator = 0
    denominator = 0

    for similar_user_id, similarity in top_similar_users.items():
        similar_user_idx = user_to_index[similar_user_id]
        rating = user_item_matrix.iloc[similar_user_idx, movie_idx]
        user_mean = user_means[similar_user_id]
        numerator += similarity * (rating - user_mean)
        denominator += abs(similarity)

    if denominator == 0:
        return user_means[user_id]

    predicted_rating = user_means[user_id] + numerator / denominator
    # Clamp rating to valid range
    return max(0.5, min(5.0, predicted_rating))


def collaborative_filtering_recommendations(user_id, movies_filtered,
                                            user_to_index, user_item_matrix,
                                            user_means, movie_to_index,
                                            user_similarity_df,
                                            get_similar_users_func,
                                            n=10, k=10):
    """
    Get top N movie recommendations for a user using collaborative filtering.
    """
    if user_id not in user_to_index:
        # Return empty dataframe with columns
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

    user_idx = user_to_index[user_id]

    # Movies not rated by the user
    user_ratings = user_item_matrix.iloc[user_idx]
    unrated_movies = user_ratings[user_ratings == 0].index

    predictions = []
    for movie_id in unrated_movies[:500]:  # limit to 500 for speed
        pred_rating = predict_rating(user_id, user_to_index, user_means,
                                     user_item_matrix, get_similar_users_func,
                                     movie_to_index, movie_id, user_similarity_df, k=k)
        predictions.append((movie_id, pred_rating))

    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, _ in predictions[:n]]

    # Merge with metadata
    recommendations = movies_filtered[movies_filtered['movieId'].isin(top_movie_ids)][
        ['movieId', 'title', 'genres']
    ]
    return recommendations
