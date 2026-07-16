# 📚 Light Novel Recommendation System

A content-based Light Novel Recommendation System built using:

- Python
- Streamlit
- TF-IDF
- Cosine Similarity
- Scikit-learn


## 🚀 Features

- Search novels using:
  - Title
  - Genres
  - Tags
  - Description

- Recommendation using TF-IDF + Cosine Similarity

- Filters:
  - Language
  - Minimum Rating
  - Number of Results
  - Exclude Yaoi
  - Exclude Boys Love
  - Exclude Shounen Ai

- Random Recommendation button

- Modern dark Streamlit interface

- Shows:
  - Title
  - Author
  - Rating
  - Genres
  - Tags
  - Language
  - Similarity %
  - Status


## 📂 Project Structure

```
LightNovel-Recommender/
│
├── app.py
├── LightNovel_Recommender.ipynb
├── requirements.txt
├── README.md
└── wn.csv
```


## ⚙ Installation

Install required libraries:

```bash
pip install -r requirements.txt
```


## ▶ Run Application

Start Streamlit:

```bash
streamlit run app.py
```


## 📓 Notebook

`LightNovel_Recommender.ipynb` contains:

- Dataset loading
- Data cleaning
- TF-IDF vectorization
- Cosine similarity
- Recommendation testing


## 🧠 How It Works

1. Light novel information is loaded from `wn.csv`.

2. Text from:
   - Title
   - Genres
   - Tags
   - Description

   is combined.

3. TF-IDF converts text into numerical vectors.

4. Cosine Similarity finds similar novels.

5. The system ranks novels based on similarity score.


## 📊 Dataset

The dataset file should be:

```
wn.csv
```

Required columns:

- title
- authors
- genres
- tags
- language
- rating
- status_coo
- description


## 🛠 Technologies

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| Scikit-learn | Machine Learning |
| TF-IDF | Text Feature Extraction |
| Cosine Similarity | Recommendation Ranking |
| Streamlit | User Interface |


## 👨‍💻 Author

Light Novel Recommendation System Project


## 📄 License

For educational purposes.