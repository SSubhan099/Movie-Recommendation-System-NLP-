from flask import Flask, render_template, request
import pandas as pd
import requests
import os
import time

from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# TMDB SETUP
# =========================================================

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"


# =========================================================
# REQUEST SESSION
# =========================================================

session = requests.Session()


# =========================================================
# LOAD DATASET
# =========================================================

movies = pd.read_csv("Movies.csv")


# =========================================================
# CREATE CONTENT
# =========================================================

movies["content"] = (
    movies["genre"].fillna("") + " " +
    movies["keywords"].fillna("") + " " +
    movies["overview"].fillna("")
)


# =========================================================
# TF-IDF
# =========================================================

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    movies["content"]
)


# =========================================================
# COSINE SIMILARITY
# =========================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_movies(movie_title, number_of_recommendations=5):

    movie_index = movies[
        movies["title"].str.lower() ==
        movie_title.lower()
    ].index

    if len(movie_index) == 0:
        return []

    movie_index = movie_index[0]

    similarity_scores = list(
        enumerate(
            similarity_matrix[movie_index]
        )
    )

    similarity_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores:

        if index == movie_index:
            continue

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "genre": movies.iloc[index]["genre"],
            "overview": movies.iloc[index]["overview"],
            "score": round(score * 100, 1),
            "poster": None
        })

        if len(recommendations) >= number_of_recommendations:
            break

    return recommendations


# =========================================================
# TMDB REQUEST WITH RETRY
# =========================================================

def tmdb_request(endpoint, params):

    if not TMDB_API_KEY:
        print("ERROR: TMDB_API_KEY not found in .env")
        return None

    params["api_key"] = TMDB_API_KEY

    for attempt in range(3):

        try:

            response = session.get(
                TMDB_BASE_URL + endpoint,
                params=params,
                timeout=15
            )

            print(
                "TMDB:",
                endpoint,
                "->",
                response.status_code
            )

            # ---------------------------------------------
            # SUCCESS
            # ---------------------------------------------

            if response.status_code == 200:

                return response.json()


            # ---------------------------------------------
            # RATE LIMIT
            # ---------------------------------------------

            if response.status_code == 429:

                print("TMDB rate limit. Retrying...")

                time.sleep(1.5)

                continue


            # ---------------------------------------------
            # OTHER ERROR
            # ---------------------------------------------

            print(
                "TMDB Error:",
                response.status_code,
                response.text[:200]
            )

            time.sleep(0.5)

        except requests.RequestException as error:

            print(
                "TMDB connection error:",
                error
            )

            time.sleep(1)


    return None


# =========================================================
# SEARCH MOVIE ON TMDB
# =========================================================

def search_tmdb_movie(movie_title):

    data = tmdb_request(
        "/search/movie",
        {
            "query": movie_title,
            "language": "en-US",
            "include_adult": False
        }
    )

    if not data:
        return None

    results = data.get(
        "results",
        []
    )

    if not results:

        print(
            "No TMDB result:",
            movie_title
        )

        return None


    requested_title = (
        movie_title
        .strip()
        .lower()
    )


    # =====================================================
    # 1. EXACT TITLE WITH POSTER
    # =====================================================

    for movie in results:

        tmdb_title = (
            movie.get("title", "")
            .strip()
            .lower()
        )

        if (
            tmdb_title == requested_title
            and movie.get("poster_path")
        ):

            return movie


    # =====================================================
    # 2. EXACT TITLE
    # =====================================================

    for movie in results:

        tmdb_title = (
            movie.get("title", "")
            .strip()
            .lower()
        )

        if tmdb_title == requested_title:

            return movie


    # =====================================================
    # 3. ANY RESULT WITH POSTER
    # =====================================================

    for movie in results:

        if movie.get("poster_path"):

            return movie


    # =====================================================
    # 4. FIRST RESULT
    # =====================================================

    return results[0]


# =========================================================
# GET MOVIE DETAILS
# =========================================================

def get_tmdb_movie_details(movie_title):

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------

    search_result = search_tmdb_movie(
        movie_title
    )

    if not search_result:
        return None


    movie_id = search_result.get(
        "id"
    )

    if not movie_id:

        return search_result


    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    details = tmdb_request(
        f"/movie/{movie_id}",
        {
            "language": "en-US"
        }
    )


    # -----------------------------------------------------
    # IMPORTANT:
    # If details request fails, KEEP SEARCH RESULT
    # because search result already contains poster_path.
    # -----------------------------------------------------

    if not details:

        return search_result


    # -----------------------------------------------------
    # If details doesn't contain poster,
    # use poster from search result.
    # -----------------------------------------------------

    if not details.get("poster_path"):

        details["poster_path"] = (
            search_result.get("poster_path")
        )


    # -----------------------------------------------------
    # If details doesn't contain title
    # use search title.
    # -----------------------------------------------------

    if not details.get("title"):

        details["title"] = (
            search_result.get("title")
        )


    return details


# =========================================================
# FORMAT MOVIE DATA
# =========================================================

def format_movie_details(movie_data):

    if not movie_data:
        return None


    # =====================================================
    # POSTER
    # =====================================================

    poster_path = movie_data.get(
        "poster_path"
    )


    if poster_path:

        poster_url = (
            TMDB_IMAGE_URL +
            poster_path
        )

    else:

        poster_url = None


    # =====================================================
    # GENRES
    # =====================================================

    genres = movie_data.get(
        "genres",
        []
    )


    genre_names = [
        genre.get("name")
        for genre in genres
        if genre.get("name")
    ]


    # =====================================================
    # RATING
    # =====================================================

    rating = movie_data.get(
        "vote_average"
    )


    # =====================================================
    # RUNTIME
    # =====================================================

    runtime = movie_data.get(
        "runtime"
    )


    # =====================================================
    # RELEASE DATE
    # =====================================================

    release_date = movie_data.get(
        "release_date"
    )


    # =====================================================
    # OVERVIEW
    # =====================================================

    overview = movie_data.get(
        "overview"
    )


    return {

        "title":
            movie_data.get(
                "title",
                "Unknown"
            ),

        "poster":
            poster_url,

        "genres":
            genre_names,

        "overview":
            overview
            if overview
            else "No overview available.",

        "release_date":
            release_date
            if release_date
            else "N/A",

        "runtime":
            runtime,

        "rating":
            rating

    }


# =========================================================
# HOME
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    selected_movie = None

    selected_movie_info = None

    recommendations = []


    # =====================================================
    # POST
    # =====================================================

    if request.method == "POST":

        selected_movie = request.form.get(
            "movie"
        )


        # =================================================
        # SELECTED MOVIE
        # =================================================

        tmdb_selected = (
            get_tmdb_movie_details(
                selected_movie
            )
        )


        selected_movie_info = (
            format_movie_details(
                tmdb_selected
            )
        )


        # =================================================
        # RECOMMENDATIONS
        # =================================================

        recommendations = (
            recommend_movies(
                selected_movie,
                5
            )
        )


        # =================================================
        # TMDB DATA FOR RECOMMENDATIONS
        # =================================================

        for movie in recommendations:

            print(
                "Fetching poster:",
                movie["title"]
            )


            tmdb_data = (
                get_tmdb_movie_details(
                    movie["title"]
                )
            )


            formatted_data = (
                format_movie_details(
                    tmdb_data
                )
            )


            if formatted_data:

                movie["poster"] = (
                    formatted_data["poster"]
                )

                movie["tmdb_genres"] = (
                    formatted_data["genres"]
                )

            else:

                movie["poster"] = None

                movie["tmdb_genres"] = []


            print(
                "Poster:",
                movie["title"],
                "=>",
                movie["poster"]
            )


    # =====================================================
    # DROPDOWN
    # =====================================================

    movie_list = movies[
        "title"
    ].tolist()


    # =====================================================
    # RENDER
    # =====================================================

    return render_template(

        "index.html",

        movies=movie_list,

        selected_movie=selected_movie,

        selected_movie_info=selected_movie_info,

        recommendations=recommendations
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )