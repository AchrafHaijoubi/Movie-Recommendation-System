# Movie Recommendation System

A web-based movie recommendation system built with Streamlit, utilizing both content-based and collaborative filtering techniques to suggest movies based on user preferences.

## Features

- **Content-Based Filtering**: Recommends movies similar to a selected movie based on genres and other content features.
- **Collaborative Filtering**: Provides personalized recommendations based on user ratings and similarities with other users.
- **Interactive UI**: Simple and intuitive interface powered by Streamlit.
- **Movie Posters**: Fetches and displays movie posters using the TMDB API.
- **Data-Driven**: Uses the MovieLens dataset for ratings and movie metadata.

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── recommendations.py              # Recommendation algorithms (content-based and collaborative filtering)
├── collaborative_filtering.ipynb   # Notebook for collaborative filtering analysis
├── EDA.ipynb                       # Exploratory Data Analysis notebook
├── item_content_based_filtering.ipynb  # Notebook for content-based filtering
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── data/                           # Data files (not included in repo)
    ├── ml-32m/                     # MovieLens 32M dataset
    ├── movies_filtered.csv         # Filtered movies data
    ├── content_matrix.csv          # Content features matrix
    ├── small_user_item_matrix.csv  # User-item matrix for collaborative filtering
    └── small_user_similarity.csv   # User similarity matrix
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your TMDB API key:
     ```
     TMDB_API_KEY=your_tmdb_api_key_here
     ```
   - You can get a TMDB API key from [The Movie Database](https://www.themoviedb.org/settings/api).

4. Prepare the data:
   - Download the MovieLens 32M dataset from [MovieLens](https://grouplens.org/datasets/movielens/32m/).
   - Place the `links.csv` file in a `ml-32m/` directory.
   - Ensure the following preprocessed files are available (generated from the notebooks):
     - `movies_filtered.csv`
     - `content_matrix.csv`
     - `small_user_item_matrix.csv`
     - `small_user_similarity.csv`

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`).

3. Select a recommendation method:
   - **Content-Based**: Choose a movie from the dropdown and specify the number of recommendations.
   - **Collaborative Filtering**: Enter a MovieLens User ID to get personalized recommendations.

4. Click "Recommend movies" to see suggestions with posters and genres.

## Data Preparation

The system relies on preprocessed data. Use the provided Jupyter notebooks to generate the necessary files:

- `EDA.ipynb`: Perform exploratory data analysis on the MovieLens dataset.
- `item_content_based_filtering.ipynb`: Create content features and clustering for content-based recommendations.
- `collaborative_filtering.ipynb`: Compute user similarities and matrices for collaborative filtering.

Ensure all data files are in the correct locations as referenced in `app.py`.

## Dependencies

- numpy
- pandas
- scikit-learn
- jupyterlab
- streamlit
- requests
- python-dotenv

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- MovieLens dataset by GroupLens Research.
- TMDB API for movie posters and details.
- Streamlit for the web interface.