import streamlit as st
import pandas as pd
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from googleapiclient.discovery import build
import numpy as np

# ── Configuration de la page ──────────────────────────
st.set_page_config(
    page_title="CINÉCREUSE",
    page_icon="🎬",
    layout="wide"
)

# ── YouTube API ───────────────────────────────────────
YOUTUBE_API_KEY = 'AIzaSyDkKZie4PBfLPnKsfo9hSjMluJOYjNLoS8'

def get_trailer(titre):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            q=f"{titre} trailer français",
            part='snippet',
            maxResults=1,
            type='video'
        )
        response = request.execute()
        if response['items']:
            video_id = response['items'][0]['id']['videoId']
            return f"https://www.youtube.com/embed/{video_id}"
        return None
    except:
        return None

# ── Charger les données + ML ──────────────────────────
@st.cache_data
def charger_données():
    df = pd.read_csv('dataset_final_tmdb_v2.1.csv', low_memory=False)
    df_films = df.drop_duplicates(subset=['tconst'])[[
        'tconst', 'primaryTitle', 'titre', 'startYear', 'genres',
        'averageRating', 'numVotes', 'overview',
        'poster_path', 'budget', 'revenue',
        'production_countries', 'runtimeMinutes'
    ]].reset_index(drop=True)
    return df_films

@st.cache_resource
def construire_modele(df_films):
    df_films = df_films.copy()
    df_films['genres'] = df_films['genres'].fillna('')
    df_films['overview'] = df_films['overview'].fillna('')
    df_films['startYear'] = pd.to_numeric(
        df_films['startYear'], errors='coerce').fillna(0)
    df_films['contenu'] = (
        df_films['genres'] + ' ' +
        df_films['genres'] + ' ' +
        df_films['genres'] + ' ' +
        df_films['genres'] + ' ' +
        df_films['overview']
    )
    tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_films['contenu'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scaler = MinMaxScaler()
    df_films['note_norm'] = scaler.fit_transform(
        df_films[['averageRating']].fillna(0))
    return cosine_sim, df_films

df_films = charger_données()
cosine_sim, df_ml = construire_modele(df_films)

# ── Initialiser session_state ─────────────────────────
for key in ['film_aleatoire', 'film_drama', 'film_comedie', 'film_action', 'page']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'films_aleatoires' not in st.session_state:
    st.session_state['films_aleatoires'] = df_films.dropna(
        subset=['poster_path']).sample(10).reset_index(drop=True)

if 'favoris' not in st.session_state:
    st.session_state['favoris'] = []

# ── Fonction recommander ──────────────────────────────
def recommander(titre, n=5):
    resultats = df_ml[
        df_ml['titre'].str.contains(titre, case=False, na=False) |
        df_ml['primaryTitle'].str.contains(titre, case=False, na=False)
    ]
    if resultats.empty:
        return None
    film = resultats.iloc[0]
    idx = film.name
    scores_genres = cosine_sim[idx]
    scores_notes = df_ml['note_norm'].values
    scores_final = 0.8 * scores_genres + 0.2 * scores_notes
    scores = list(enumerate(scores_final))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]
    indices = [i[0] for i in scores]
    return df_ml.iloc[indices]

# ── Fonction afficher détails ─────────────────────────
def afficher_details(film, key_prefix):
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(f"https://image.tmdb.org/t/p/w500{film['poster_path']}", width=150)
        if st.button("🤍 Favoris", key=f"fav_{key_prefix}_{film['tconst']}"):
            if film['tconst'] not in st.session_state['favoris']:
                st.session_state['favoris'].append(film['tconst'])
                st.success("Ajouté aux favoris!")
    with col2:
        st.markdown(f"### {film['titre']} ({int(film['startYear'])})")
        st.markdown(f"**Genres :** {film['genres']}")
        st.markdown(f"**Note IMDb :** ⭐ {film['averageRating']}")
        st.markdown(f"**Durée :** {int(film['runtimeMinutes'])} min")
        st.markdown(f"**Synopsis :** {film['overview']}")
        trailer_url = get_trailer(film['titre'])
        if trailer_url:
            st.markdown("**🎬 Bande-annonce :**")
            col_video, col_vide = st.columns([1, 1])
            with col_video:
                st.components.v1.iframe(trailer_url, height=250)
        if st.button("🍿 Voir les recommandations",
                     key=f"reco_{key_prefix}_{film['tconst']}"):
            recommandations = recommander(film['titre'])
            if recommandations is not None:
                cols = st.columns(5)
                for i, (_, row) in enumerate(recommandations.iterrows()):
                    with cols[i]:
                        st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}",
                                 use_container_width=True)
                        with st.expander(f"**{row['titre']}** ⭐{row['averageRating']}"):
                            st.markdown(f"**Année :** {int(row['startYear'])}")
                            st.markdown(f"**Genres :** {row['genres']}")
                            st.markdown(f"**Synopsis :** {row['overview']}")
    st.divider()

# ── Fonction afficher catégorie ───────────────────────
def afficher_categorie(titre_section, films, key_prefix, session_key):
    st.markdown(f"""
        <h3 style="
            font-size: 35px;
            background: linear-gradient(45deg, #ff4444, #ff6b35, #ff9500, #ffcc00, #fff000);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">{titre_section}</h3>
    """, unsafe_allow_html=True)
    cols = st.columns(10)
    for i, (_, row) in enumerate(films.iterrows()):
        with cols[i % 10]:
            st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}",
                     use_container_width=True)
            if st.button(f"{row['titre']} ⭐{row['averageRating']}",
                         key=f"{key_prefix}_{i}_{row['tconst']}",
                         use_container_width=True):
                st.session_state[session_key] = row['tconst']
    if session_key in st.session_state and st.session_state[session_key] is not None:
        film = df_films[df_films['tconst'] == st.session_state[session_key]].iloc[0]
        afficher_details(film, key_prefix)

# ── CSS ───────────────────────────────────────────────
with open("logo3.png", "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()
with open("20.jpg", "rb") as f:
    fondo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <style>
    .stApp {{ background-color: #606060; }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: url("data:image/jpg;base64,{fondo_base64}");
        background-size: 500px 500px;
        background-position: center;
        background-attachment: fixed;
        opacity: 0.05;
        z-index: 0;
    }}
    .stApp > * {{ position: relative; z-index: 1; }}
    [data-testid="stImage"] img {{
        border-radius: 20px !important;
        object-fit: cover;
    }}
    iframe {{
        border-radius: 20px !important;
        overflow: hidden;
    }}
    .logo-circulaire {{
        width: 150px; height: 150px;
        border-radius: 50%;
        object-fit: cover;
        animation: girar 8s linear infinite;
    }}
    @keyframes girar {{
        from {{ transform: rotate(0deg); }}
        to   {{ transform: rotate(360deg); }}
    }}
    [data-testid="stTextInput"] input {{
        background-color: rgba(0, 0, 0, 0.0) !important;
        border: 1px solid rgba(255, 255, 255, 0.55) !important;
        color: white !important;
    }}
    [data-testid="stTextInput"] > div {{
        background-color: rgba(0, 0, 0, 0.0) !important;
        border: none !important;
    }}
    [data-testid="stTextInput"] > div > div {{
        background-color: rgba(0, 0, 0, 0.0) !important;
    }}
    [data-testid="stTextInput"] input::placeholder {{
        color: rgba(255, 255, 255, 0.55) !important;
    }}
    [data-testid="stSelectbox"] > div > div {{
        background-color: rgba(0, 0, 0, 0.0) !important;
        border: 1px solid rgba(255, 255, 255, 0.55) !important;
        color: white !important;
    }}
    [data-testid="stSelectbox"] span {{
        color: white !important;
    }}
    [data-testid="stSelectbox"] {{
        cursor: pointer !important;
    }}
    [data-testid="stSelectbox"] * {{
        cursor: pointer !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
        min-width: 80px !important;
        max-width: 80px !important;
    }}
    [data-testid="stSidebar"] button {{
        background-color: transparent !important;
        border: none !important;
        font-size: 28px !important;
        width: 100% !important;
        padding: 15px 0 !important;
    }}
    [data-testid="stSidebar"] button:hover {{
        background-color: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    if st.button("🏠", help="Accueil", use_container_width=True):
        st.session_state['page'] = 'accueil'
    if st.button("🤍", help="Favoris", use_container_width=True):
        st.session_state['page'] = 'favoris'
    if st.button("🔔", help="Notifications", use_container_width=True):
        st.session_state['page'] = 'notifications'
    if st.button("👤", help="Profil", use_container_width=True):
        st.session_state['page'] = 'profil'

# ── Page Favoris ──────────────────────────────────────
if st.session_state['page'] == 'favoris':
    st.markdown("### 🤍 Mes Favoris")
    if st.session_state['favoris']:
        films_favoris = df_films[df_films['tconst'].isin(st.session_state['favoris'])]
        for _, film in films_favoris.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(f"https://image.tmdb.org/t/p/w500{film['poster_path']}", width=150)
            with col2:
                st.markdown(f"### {film['titre']} ({int(film['startYear'])})")
                st.markdown(f"**Genres :** {film['genres']}")
                st.markdown(f"**Note IMDb :** ⭐ {film['averageRating']}")
            st.divider()
    else:
        st.info("Aucun favori. Ajoutez des films depuis l'accueil!")

# ── Page Notifications ────────────────────────────────
elif st.session_state['page'] == 'notifications':
    st.markdown("### 🔔 Notifications")
    st.info("Pas de nouvelles notifications.")

# ── Page Profil ───────────────────────────────────────
elif st.session_state['page'] == 'profil':
    st.markdown("### 👤 Mon Profil")
    st.markdown("**Nom :** Utilisateur CinéCreuse")
    st.markdown(f"**Favoris :** {len(st.session_state['favoris'])} films")

# ── Page Accueil ──────────────────────────────────────
else:
    # ── En-tête ───────────────────────────────────────
    col_logo, col_titre = st.columns([1, 8])
    with col_logo:
        st.markdown(f"""
            <img src="data:image/png;base64,{logo_base64}" class="logo-circulaire">
        """, unsafe_allow_html=True)
    with col_titre:
        st.markdown("""
            <h1 style="
                font-size: 90px;
                font-weight: 900;
                letter-spacing: 6px;
                background: linear-gradient(45deg, #c0392b, #e74c3c, #e67e22, #f39c12, #f1c40f);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0;
            ">CINÉCREUSE</h1>
        """, unsafe_allow_html=True)

    # ── Barre de navigation ───────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_search, col_genre, col_pays = st.columns([3, 1, 1])
    with col_search:
        film_input = st.text_input("", placeholder="🔍 Rechercher un film...",
                                    label_visibility="collapsed")
    with col_genre:
        tous_genres = sorted(df_films['genres'].dropna().str.split(',')
                            .explode().str.strip().unique().tolist())
        tous_genres.insert(0, "🎭 Tous les genres")
        genre_selectionne = st.selectbox("", tous_genres,
                                          label_visibility="collapsed")
    with col_pays:
        tous_pays = sorted(df_films['production_countries'].dropna()
                          .str.replace("'", "").str.replace("[", "")
                          .str.replace("]", "").str.split(",")
                          .explode().str.strip().unique().tolist())
        tous_pays = [p for p in tous_pays if p != '']
        tous_pays.insert(0, "🌍 Tous les pays")
        pays_selectionne = st.selectbox("", tous_pays,
                                         label_visibility="collapsed")

    # ── Filtrer ───────────────────────────────────────
    df_filtre = df_films.copy()
    if genre_selectionne != "🎭 Tous les genres":
        df_filtre = df_filtre[df_filtre['genres'].str.contains(
                              genre_selectionne, na=False)]
    if pays_selectionne != "🌍 Tous les pays":
        df_filtre = df_filtre[df_filtre['production_countries'].str.contains(
                              pays_selectionne, na=False)]

    # ── Résultats ─────────────────────────────────────
    if film_input:
        resultats = df_filtre[
            df_filtre['titre'].str.contains(film_input, case=False, na=False) |
            df_filtre['primaryTitle'].str.contains(film_input, case=False, na=False)
        ]
        if not resultats.empty:
            st.success(f"{len(resultats)} film(s) trouvé(s) pour : **{film_input}**")
            st.markdown("### 🎬 Résultats de recherche")
            for _, film in resultats.iterrows():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(f"https://image.tmdb.org/t/p/w500{film['poster_path']}", width=150)
                with col2:
                    st.markdown(f"### {film['titre']} ({int(film['startYear'])})")
                    st.markdown(f"**Genres :** {film['genres']}")
                    st.markdown(f"**Note IMDb :** ⭐ {film['averageRating']}")
                    st.markdown(f"**Durée :** {int(film['runtimeMinutes'])} min")
                    st.markdown(f"**Synopsis :** {film['overview']}")
                    trailer_url = get_trailer(film['titre'])
                    if trailer_url:
                        st.markdown("**🎬 Bande-annonce :**")
                        st.components.v1.iframe(trailer_url, height=450)
                    if st.button("🍿 Voir les recommandations",
                                 key=f"reco_recherche_{film['tconst']}"):
                        recommandations = recommander(film['titre'])
                        if recommandations is not None:
                            cols = st.columns(5)
                            for i, (_, row) in enumerate(recommandations.iterrows()):
                                with cols[i]:
                                    st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}",
                                             use_container_width=True)
                                    with st.expander(f"**{row['titre']}** ⭐{row['averageRating']}"):
                                        st.markdown(f"**Année :** {int(row['startYear'])}")
                                        st.markdown(f"**Genres :** {row['genres']}")
                                        st.markdown(f"**Synopsis :** {row['overview']}")
                st.divider()
        else:
            st.warning("⚠️ Film non trouvé. Essayez un autre titre.")

    elif genre_selectionne != "🎭 Tous les genres" or pays_selectionne != "🌍 Tous les pays":
        resultats = df_filtre.sort_values('averageRating', ascending=False).head(20)
        if not resultats.empty:
            st.markdown("### 🎬 Résultats")
            for _, film in resultats.iterrows():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(f"https://image.tmdb.org/t/p/w500{film['poster_path']}", width=150)
                with col2:
                    st.markdown(f"### {film['titre']} ({int(film['startYear'])})")
                    st.markdown(f"**Genres :** {film['genres']}")
                    st.markdown(f"**Note IMDb :** ⭐ {film['averageRating']}")
                    st.markdown(f"**Durée :** {int(film['runtimeMinutes'])} min")
                    st.markdown(f"**Synopsis :** {film['overview']}")
                    if st.button("🍿 Voir les recommandations",
                                 key=f"reco_filtre_{film['tconst']}"):
                        recommandations = recommander(film['titre'])
                        if recommandations is not None:
                            cols = st.columns(5)
                            for i, (_, row) in enumerate(recommandations.iterrows()):
                                with cols[i]:
                                    st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}",
                                             use_container_width=True)
                                    with st.expander(f"**{row['titre']}** ⭐{row['averageRating']}"):
                                        st.markdown(f"**Année :** {int(row['startYear'])}")
                                        st.markdown(f"**Genres :** {row['genres']}")
                                        st.markdown(f"**Synopsis :** {row['overview']}")
                st.divider()

    # ── Catégories ────────────────────────────────────
    afficher_categorie(
        "🎲 Films à découvrir aujourd'hui",
        st.session_state['films_aleatoires'],
        "aleatoire",
        "film_aleatoire"
    )

    top_drama = df_films[df_films['genres'].str.contains('Drama', na=False)]\
        .sort_values('averageRating', ascending=False).head(10)
    afficher_categorie("🏆 Top Drama", top_drama, "drama", "film_drama")

    top_comedie = df_films[df_films['genres'].str.contains('Comedy', na=False)]\
        .sort_values('averageRating', ascending=False).head(10)
    afficher_categorie("🎭 Top Comédie", top_comedie, "comedie", "film_comedie")

    top_action = df_films[df_films['genres'].str.contains('Action', na=False)]\
        .sort_values('averageRating', ascending=False).head(10)
    afficher_categorie("🚀 Top Action", top_action, "action", "film_action")

# ── Pied de page ──────────────────────────────────────
st.divider()
st.markdown("<center>Wild Code School 2026 — Projet 2</center>",
            unsafe_allow_html=True)