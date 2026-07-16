# ==========================================================
# LIGHT NOVEL RECOMMENDER SYSTEM
# Streamlit Application
# TF-IDF + Cosine Similarity
# ==========================================================


import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Light Novel Recommender",
    page_icon="📚",
    layout="wide"
)



# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
"""
<style>

body{
background:#0e1117;
}

.main{
background:#0e1117;
color:white;
}


.title{

text-align:center;
font-size:42px;
font-weight:bold;

}


.card{

background:#1c1f26;
padding:20px;
border-radius:15px;

}


.similarity{

color:#00ff99;
font-weight:bold;

}


</style>
""",
unsafe_allow_html=True
)



# ==========================================================
# LOAD DATA
# ==========================================================


@st.cache_data
def load_data():


    df = pd.read_csv(
        "wn.csv",
        engine="python",
        on_bad_lines="skip"
    )


    required = [

        "title",
        "authors",
        "genres",
        "tags",
        "language",
        "rating",
        "status_coo",
        "description"

    ]



    for col in required:

        if col in df.columns:

            df[col]=df[col].fillna("")



    # ------------------------------
    # REMOVE UNWANTED GENRES
    # ------------------------------


    blocked = [

        "yaoi",
        "boys love",
        "harem"

    ]


    pattern="|".join(blocked)



    df=df[
        ~df["genres"]
        .astype(str)
        .str.lower()
        .str.contains(
            pattern,
            na=False
        )
    ]



    df=df.reset_index(drop=True)



    # ------------------------------
    # FIX RATING
    # ------------------------------


    def extract_rating(value):


        nums=re.findall(
            r"\d+\.\d+|\d+",
            str(value)
        )


        if nums:


            rating=float(nums[0])


            if rating<=5:

                return rating


        return 0



    df["rating"]=df["rating"].apply(
        extract_rating
    )



    # ------------------------------
    # CREATE SEARCH TEXT
    # ------------------------------


    df["combined"]=(

        df["title"].astype(str)
        +" "
        +
        df["genres"].astype(str)
        +" "
        +
        df["tags"].astype(str)
        +" "
        +
        df["description"].astype(str)

    )


    return df



df=load_data()



# ==========================================================
# TEXT CLEANING
# ==========================================================


def clean(text):

    text=str(text).lower()

    text=re.sub(
        r"[^a-z0-9 ]",
        " ",
        text
    )

    return text



df["combined"]=df["combined"].apply(clean)

# ==========================================================
# TF-IDF MODEL
# ==========================================================


@st.cache_resource
def create_model(data):


    vectorizer = TfidfVectorizer(

        stop_words="english",

        max_features=10000,

        ngram_range=(1,2)

    )


    matrix = vectorizer.fit_transform(
        data["combined"]
    )


    return vectorizer, matrix



vectorizer, tfidf_matrix = create_model(df)



# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


def clean_list(value):


    if isinstance(
        value,
        (list, np.ndarray)
    ):

        return ", ".join(
            map(str,value)
        )


    return str(value)




# ==========================================================
# RECOMMENDATION FUNCTION
# ==========================================================


def recommend(
        query,
        language="Any",
        minimum_rating=0,
        top_n=5
):


    query = clean(query)



    query_vector = vectorizer.transform(
        [query]
    )



    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()



    result = df.copy()


    result["similarity"] = similarity



    # ------------------------------
    # LANGUAGE FILTER
    # ------------------------------

    if language != "Any":


        result = result[
            result["language"]
            .astype(str)
            .str.lower()
            .str.contains(
                language.lower(),
                na=False
            )
        ]



    # ------------------------------
    # RATING FILTER
    # ------------------------------

    if minimum_rating > 0:


        result = result[
            result["rating"]
            >= minimum_rating
        ]



    # ------------------------------
    # SORT RESULTS
    # ------------------------------

    result = result.sort_values(

        by="similarity",

        ascending=False

    )



    # Remove zero similarity

    result = result[
        result["similarity"] > 0
    ]



    return result.head(top_n)




# ==========================================================
# RANDOM RECOMMENDATION
# ==========================================================


def random_recommend(
        language="Any",
        minimum_rating=0
):


    result=df.copy()



    if language!="Any":


        result=result[
            result["language"]
            .astype(str)
            .str.lower()
            .str.contains(
                language.lower(),
                na=False
            )
        ]



    if minimum_rating>0:


        result=result[
            result["rating"]
            >= minimum_rating
        ]



    if result.empty:

        return None



    return result.sample(1).iloc[0]

# ==========================================================
# STREAMLIT USER INTERFACE
# ==========================================================


st.markdown(
"""
<div class="title">
📚 Light Novel Recommendation System
</div>
""",
unsafe_allow_html=True
)


st.write(
"Search light novels using Title, Genres, Tags and Description"
)



# ==========================================================
# SIDEBAR FILTERS
# ==========================================================


st.sidebar.header(
    "⚙ Recommendation Settings"
)



# Languages

languages = [

    "Any"

] + sorted(
    df["language"]
    .astype(str)
    .unique()
)



language = st.sidebar.selectbox(
    "🌍 Language",
    languages
)



# Rating

minimum_rating = st.sidebar.slider(

    "⭐ Minimum Rating",

    0.0,

    5.0,

    0.0,

    0.1

)



# Number of results

number_results = st.sidebar.slider(

    "📚 Number of Results",

    1,

    20,

    5

)



st.sidebar.markdown("---")


# ==========================================================
# SEARCH BOX
# ==========================================================


query = st.text_input(
    "🔍 Search Light Novel",
    placeholder="Example: regression apocalypse cultivation fantasy"
)



col1, col2 = st.columns(2)



with col1:

    search_button = st.button(
        "🔍 Search"
    )



with col2:

    random_button = st.button(
        "🎲 Random Recommendation"
    )



# ==========================================================
# SEARCH RESULTS
# ==========================================================


if search_button:


    if query.strip()=="":


        st.warning(
            "Please enter a search query"
        )


    else:


        results = recommend(

            query,

            language,

            minimum_rating,

            number_results

        )



        if results.empty:


            st.error(
                "No novels found"
            )


        else:


            st.success(
                f"{len(results)} recommendations found"
            )



            for _,row in results.iterrows():



                with st.expander(

                    "📖 "
                    +
                    str(row["title"])

                ):



                    st.write(
                        "⭐ Rating:",
                        row["rating"]
                    )


                    st.write(
                        "👤 Author:",
                        clean_list(
                            row["authors"]
                        )
                    )


                    st.write(
                        "📚 Genres:",
                        clean_list(
                            row["genres"]
                        )
                    )


                    st.write(
                        "🏷 Tags:",
                        clean_list(
                            row["tags"]
                        )
                    )


                    st.write(
                        "🌍 Language:",
                        row["language"]
                    )


                    st.write(
                        "📖 Status:",
                        row["status_coo"]
                    )



                    similarity = row["similarity"]



                    st.markdown(

                    f"""
                    <p class="similarity">
                    🎯 Similarity:
                    {similarity*100:.2f}%
                    </p>
                    """,

                    unsafe_allow_html=True

                    )



                    st.progress(
                        float(similarity)
                    )



                    if row["description"]:


                        st.write(
                            "📝 Description:"
                        )


                        st.write(
                            row["description"]
                        )




# ==========================================================
# RANDOM RESULT
# ==========================================================


if random_button:


    novel = random_recommend(

        language,

        minimum_rating

    )



    if novel is None:


        st.error(
            "No novels available"
        )



    else:


        st.success(
            "🎲 Random Recommendation"
        )



        with st.expander(

            "📖 "
            +
            str(novel["title"]),

            expanded=True

        ):



            st.write(
                "⭐ Rating:",
                novel["rating"]
            )


            st.write(
                "👤 Author:",
                clean_list(
                    novel["authors"]
                )
            )


            st.write(
                "📚 Genres:",
                clean_list(
                    novel["genres"]
                )
            )


            st.write(
                "🏷 Tags:",
                clean_list(
                    novel["tags"]
                )
            )


            st.write(
                "🌍 Language:",
                novel["language"]
            )


            st.write(
                "📖 Status:",
                novel["status_coo"]
            )



# ==========================================================
# FOOTER
# ==========================================================


st.markdown("---")


st.caption(
"Built with Streamlit • TF-IDF • Cosine Similarity • Scikit-Learn"
)