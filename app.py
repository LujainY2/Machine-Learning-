import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# Load data from pickle files
movies = pickle.load(open("movies11.pkl", 'rb'))
popular_movies_rankings = pickle.load(open("popular_movies_ranking.pkl", 'rb'))
st.markdown(
    """
    <style>
    /* Background color for the whole page including body */
    body {
    background-color: #000000 !important; /* Entire page background */
    color: #FFFFFF; /* Default text color */
    }
    
    /* Ensure all text elements are white */
    .css-18e3th9, .css-1d391kg, .css-k1vhr4, .css-10trblm, .css-1cpxqw2, h1, h2, h3, h4, h5, h6, p, span, label {
    color: #FFFFFF !important; /* White text */
    }

    /* Change select box and input box background color */
    .stMultiSelect > div {
    background-color: #FFFFFF !important; /* Select box background */
    color: #000000 !important; /* Select box text color */
    }

    /* Change dropdown options in the select box */
    .stMultiSelect div[role="listbox"] {
    background-color: #FFFFFF !important; /* Dropdown options background */
    color: #000000 !important; /* Dropdown options text color */
    }

    /* Set a neutral border for the select box */
    .stMultiSelect .css-1cpxqw2 {
    border: 1px solid #FFFFFF!important; /* Border color */
    }

   

    /* Change opacity when hovering over the image */
    .bottom:hover {
    opacity: 0.1; /* Opacity on hover */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Add image to the bottom of the page with inline CSS for opacity
st.markdown(
    """
    <div class="bottom">
        <img src="https://i.pinimg.com/originals/d1/1c/e4/d11ce48fdf024114ae3f5d3fe2fa6be7.jpg" style="width:100%; height: 40%;opacity: 0.6;">
    </div>
    """,
    unsafe_allow_html=True
)


# Access API poster
def fetch_poster(movie_id):
    base_url = 'https://api.themoviedb.org/3/movie/'
    api_key = '2beceafe4a67fecbec3345b84003c256'
    url = f'{base_url}{movie_id}?api_key={api_key}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f'https://image.tmdb.org/t/p/w500{poster_path}'
    return 'Poster not found'


# Display top-rated popular movies
def display_top_rated_popular_movies():
    popular_movies_average_rankings = popular_movies_rankings[["title", "vote_average"]].groupby('title').mean()
    top_rated_movies = popular_movies_average_rankings.sort_values(by="vote_average", ascending=False).head(5)

    st.subheader("Top Rated Popular Movies")

    # Create columns for the top 5 movies
    cols = st.columns(len(top_rated_movies))

    for col, (title, row) in zip(cols, top_rated_movies.iterrows()):
        movie_id = movies[movies['title'] == title].id.values[0]
        poster_url = fetch_poster(movie_id)

        with col:
            st.image(poster_url, caption=f"{title} - Average Rating: {row['vote_average']:.2f}")


# Call the function to display top-rated popular movies
display_top_rated_popular_movies()

# Allow users to select movies they like
st.subheader("Select Movies You Enjoyed")
list_of_movies_enjoyed = st.multiselect("Select movies:", options=movies['title'].values)

# Ensure TF-IDF vectorizer and matrix
vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, max_features=5000)
vectorized_data = vectorizer.fit_transform(movies['tags'])
tfidf_df = pd.DataFrame.sparse.from_spmatrix(vectorized_data, index=movies['title'],
                                             columns=vectorizer.get_feature_names_out())


def generate_user_profile(list_of_movies_enjoyed, tfidf_df):
    # Ensure index uniqueness in tfidf_df
    tfidf_df = tfidf_df.loc[~tfidf_df.index.duplicated(keep='first')]

    # Create a subset of only the movies the user has enjoyed
    movies_enjoyed_df = tfidf_df.reindex(list_of_movies_enjoyed)

    # Drop any rows with NaN values in case some of the enjoyed movies are not in the DataFrame
    movies_enjoyed_df = movies_enjoyed_df.dropna()

    # Generate the user profile by finding the average scores of movies they enjoyed
    user_prof = movies_enjoyed_df.mean()

    return user_prof


# Generate user profile
if list_of_movies_enjoyed:
    user_profile = generate_user_profile(list_of_movies_enjoyed, tfidf_df)


    def recommend_movies(user_profile, tfidf_df, movies, fetch_poster):
        # Ensure index uniqueness in tfidf_df
        tfidf_df = tfidf_df.loc[~tfidf_df.index.duplicated(keep='first')]

        cosine_similarity_array = cosine_similarity(tfidf_df)
        cosine_similarity_df = pd.DataFrame(cosine_similarity_array, index=tfidf_df.index, columns=tfidf_df.index)

        # Create a DataFrame with the user profile and compute similarities
        user_profile_df = pd.DataFrame([user_profile], index=['user_profile'], columns=tfidf_df.columns)
        similarity_scores = cosine_similarity(user_profile_df, tfidf_df).flatten()

        # Create a DataFrame for similarity scores
        similarity_df = pd.DataFrame({
            'title': tfidf_df.index,
            'similarity': similarity_scores
        }).sort_values(by='similarity', ascending=False)

        # Exclude the movies already enjoyed
        similarity_df = similarity_df[~similarity_df['title'].isin(list_of_movies_enjoyed)]

        # Get the top 5 recommendations
        top_recommendations = similarity_df.head(5)

        recommend_movie = []
        recommend_poster = []
        for _, row in top_recommendations.iterrows():
            movie_id = movies[movies['title'] == row['title']].id.values[0]
            recommend_movie.append(row['title'])
            recommend_poster.append(fetch_poster(movie_id))

        return recommend_movie, recommend_poster


    # Display recommended movies based on user profile
    if st.button("Show Recommendations"):
        recommended_movies, recommended_posters = recommend_movies(user_profile, tfidf_df, movies, fetch_poster)

        st.subheader("Recommended Movies")
        cols = st.columns(len(recommended_movies))

        for col, (name, poster) in zip(cols, zip(recommended_movies, recommended_posters)):
            with col:
                st.image(poster, caption=name)
else:
    st.write("Please select some movies to get recommendations.")