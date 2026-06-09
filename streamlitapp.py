import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from googleapiclient.discovery import build
import numpy as np

st.set_page_config(page_title="CINÉCREUSE", page_icon="🎬", layout="wide")

YOUTUBE_API_KEY = 'AIzaSyDkKZie4PBfLPnKsfo9hSjMluJOYjNLoS8'

def get_titre(film):
    try:
        titre = film['titre']
        primary = film['primaryTitle']
        return titre if pd.notna(titre) and str(titre).strip() else primary
    except:
        return film['primaryTitle']

def get_trailer(titre):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(q=f"{titre} trailer français", part='snippet', maxResults=1, type='video')
        response = request.execute()
        if response['items']:
            return f"https://www.youtube.com/embed/{response['items'][0]['id']['videoId']}"
        return None
    except:
        return None

@st.cache_data
def charger_données():
    df = pd.read_csv('dataset_final_tmdb_v2.1.csv', low_memory=False)
    df_films = df.drop_duplicates(subset=['tconst'])[[
        'tconst', 'primaryTitle', 'titre', 'startYear', 'genres',
        'averageRating', 'numVotes', 'overview', 'poster_path',
        'budget', 'revenue', 'production_countries', 'runtimeMinutes'
    ]].reset_index(drop=True)
    try:
        df_acteurs = pd.read_csv('film_artiste.csv', low_memory=False)
        if 'tconst' in df_acteurs.columns and 'primaryName' in df_acteurs.columns:
            acteurs_par_film = df_acteurs.groupby('tconst')['primaryName']\
                .apply(lambda x: ', '.join(x.dropna())).reset_index()
            acteurs_par_film.columns = ['tconst', 'acteurs']
            df_films = df_films.merge(acteurs_par_film, on='tconst', how='left')
        else:
            df_films['acteurs'] = ''
    except:
        df_films['acteurs'] = ''
    return df_films

@st.cache_resource
def construire_modele(df_films):
    df_films = df_films.copy()
    df_films['genres'] = df_films['genres'].fillna('')
    df_films['overview'] = df_films['overview'].fillna('')
    df_films['production_countries'] = df_films['production_countries'].fillna('') if 'production_countries' in df_films.columns else ''
    df_films['startYear'] = pd.to_numeric(df_films['startYear'], errors='coerce').fillna(0)
    df_films['contenu'] = (
        (df_films['genres'] + ' ') * 4 +
        (df_films['production_countries'] + ' ') * 4 +
        df_films['overview']
    
    )
    tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_films['contenu'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scaler = MinMaxScaler()
    df_films['note_norm'] = scaler.fit_transform(df_films[['averageRating']].fillna(0))
    return cosine_sim, df_films

df_films = charger_données()
cosine_sim, df_ml = construire_modele(df_films)

@st.cache_data
def get_df_france(df):
    pays_a_exclure = [
        'United States', 'United Kingdom', 'Germany', 'Italy', 'Spain',
        'Japan', 'China', 'India', 'Australia', 'Canada', 'Brazil',
        'Mexico', 'Russia', 'South Korea', 'Sweden', 'Denmark',
        'Netherlands', 'Belgium', 'Switzerland', 'Austria', 'Poland'
    ]
    mask = df['production_countries'].str.contains('France', na=False)
    for pays in pays_a_exclure:
        mask = mask & ~df['production_countries'].str.contains(pays, na=False)
    return df[mask].dropna(subset=['poster_path']).reset_index(drop=True)

@st.cache_data
def get_df_classique(df):
    return df[(df['startYear'] >= 1950) & (df['startYear'] <= 1990)
    ].dropna(subset=['poster_path']).reset_index(drop=True)

@st.cache_data
def get_df_action(df):
    return df[df['genres'].str.contains('Action', na=False)
    ].dropna(subset=['poster_path']).reset_index(drop=True)

df_france = get_df_france(df_films)
df_classique = get_df_classique(df_films)
df_action = get_df_action(df_films)

for key in ['film_aleatoire', 'film_france', 'film_classique', 'film_action2', 'page']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'films_aleatoires' not in st.session_state:
    st.session_state['films_aleatoires'] = df_films.dropna(
        subset=['poster_path']).sample(10).reset_index(drop=True)

if 'cat_france' not in st.session_state:
    st.session_state['cat_france'] = df_france.sample(
        min(10, len(df_france))).reset_index(drop=True)

if 'cat_classique' not in st.session_state:
    st.session_state['cat_classique'] = df_classique.sample(
        min(10, len(df_classique))).reset_index(drop=True)

if 'cat_action2' not in st.session_state:
    st.session_state['cat_action2'] = df_action.sample(
        min(10, len(df_action))).reset_index(drop=True)

if 'favoris' not in st.session_state:
    st.session_state['favoris'] = []

def recommander(titre, n=6):
    resultats = df_ml[
        df_ml['titre'].str.contains(titre, case=False, na=False) |
        df_ml['primaryTitle'].str.contains(titre, case=False, na=False)
    ]
    if resultats.empty:
        return None
    film = resultats.iloc[0]
    idx = film.name
    pays_film = str(film['production_countries']) if pd.notna(film['production_countries']) else ''

    scores_final = 0.8 * cosine_sim[idx] + 0.2 * df_ml['note_norm'].values
    scores = sorted(enumerate(scores_final), key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx]

    pays_asiatiques = ['Japan', 'China', 'South Korea', 'India', 'Hong Kong', 'Thailand']

    if not any(p in pays_film for p in pays_asiatiques):
        scores = [
            s for s in scores
            if not any(p in str(df_ml.iloc[s[0]]['production_countries']) for p in pays_asiatiques)
        ]

    return df_ml.iloc[[i[0] for i in scores[:n]]]

def afficher_recommandations(titre, key):
    recommandations = recommander(titre)
    if recommandations is not None:
        st.markdown("#### 🎬 Films similaires")
        cols = st.columns(3)
        for i, (_, row) in enumerate(recommandations.iterrows()):
            with cols[i % 3]:
                st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}", use_container_width=True)
                with st.expander(f"**{get_titre(row)}** ⭐{row['averageRating']}"):
                    st.markdown(f"**Année :** {int(row['startYear'])}")
                    st.markdown(f"**Genres :** {row['genres']}")
                    st.markdown(f"**Synopsis :** {row['overview']}")

def afficher_details(film, key_prefix):
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(f"https://image.tmdb.org/t/p/w500{film['poster_path']}", width=150)
        if st.button("❤️ Favoris", key=f"fav_{key_prefix}_{film['tconst']}"):
            if film['tconst'] not in st.session_state['favoris']:
                st.session_state['favoris'].append(film['tconst'])
                st.success("Ajouté aux favoris!")
    with col2:
        st.markdown(f"### {get_titre(film)} ({int(film['startYear'])})")
        st.markdown(f"**Genres :** {film['genres']}")
        st.markdown(f"**Note IMDb :** ⭐ {film['averageRating']}")
        st.markdown(f"**Durée :** {int(film['runtimeMinutes'])} min")
        if 'acteurs' in film and pd.notna(film['acteurs']) and film['acteurs']:
            st.markdown(f"**Acteurs :** {film['acteurs'][:100]}...")
        st.markdown(f"**Synopsis :** {film['overview']}")
        trailer_url = get_trailer(get_titre(film))
        if trailer_url:
            st.markdown("**🎬 Bande-annonce :**")
            with st.columns([1, 1])[0]:
                st.components.v1.iframe(trailer_url, height=250)
        voir_reco = st.button("🍿 Voir les recommandations", key=f"reco_{key_prefix}_{film['tconst']}")
    if voir_reco:
        st.session_state[f"show_reco_{key_prefix}_{film['tconst']}"] = True
    if st.session_state.get(f"show_reco_{key_prefix}_{film['tconst']}", False):
        afficher_recommandations(get_titre(film), f"{key_prefix}_{film['tconst']}")
    st.divider()

def afficher_categorie(titre_section, films, key_prefix, session_key):
    st.markdown(f"""
        <h3 style="font-size:35px;background:linear-gradient(45deg,#ff4444,#ff6b35,#ff9500,#ffcc00,#fff000);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        {titre_section}</h3>""", unsafe_allow_html=True)
    cols = st.columns(10)
    for i, (_, row) in enumerate(films.iterrows()):
        with cols[i % 10]:
            st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}", use_container_width=True)
            if st.button(f"{get_titre(row)} ⭐{row['averageRating']}", key=f"{key_prefix}_{i}_{row['tconst']}", use_container_width=True):
                st.session_state[session_key] = row['tconst']
    if session_key in st.session_state and st.session_state[session_key] is not None:
        film = df_films[df_films['tconst'] == st.session_state[session_key]].iloc[0]
        afficher_details(film, key_prefix)

def afficher_resultats(resultats, key_prefix):
    for _, film in resultats.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(f"https://image.tmdb.org/t/p/w500{film['poster_path']}", width=150)
        with col2:
            st.markdown(f"### {get_titre(film)} ({int(film['startYear'])})")
            st.markdown(f"**Genres :** {film['genres']}")
            st.markdown(f"**Note IMDb :** ⭐ {film['averageRating']}")
            st.markdown(f"**Durée :** {int(film['runtimeMinutes'])} min")
            if 'acteurs' in film and pd.notna(film['acteurs']) and film['acteurs']:
                st.markdown(f"**Acteurs :** {film['acteurs'][:100]}...")
            st.markdown(f"**Synopsis :** {film['overview']}")
            trailer_url = get_trailer(get_titre(film))
            if trailer_url:
                st.markdown("**🎬 Bande-annonce :**")
                st.components.v1.iframe(trailer_url, height=450)
            voir_reco = st.button("🍿 Voir les recommandations", key=f"reco_{key_prefix}_{film['tconst']}")
        if voir_reco:
            afficher_recommandations(get_titre(film), f"{key_prefix}_{film['tconst']}")
        st.divider()

def style_graph(fig, ax, titre):
    BG = '#606060'
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_title(titre, color='white', fontsize=14, pad=15)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.2))

with open("logo3.png", "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()
with open("20.jpg", "rb") as f:
    fondo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <style>
    .stApp {{ background-color: #606060; }}
    .stApp::before {{
        content: ""; position: fixed; top: 0; left: 0;
        width: 100%; height: 100%;
        background-image: url("data:image/jpg;base64,{fondo_base64}");
        background-size: 500px 500px; background-position: center;
        background-attachment: fixed; opacity: 0.05; z-index: 0;
    }}
    .stApp > * {{ position: relative; z-index: 1; }}
    [data-testid="stImage"] img {{ border-radius: 20px !important; object-fit: cover; }}
    iframe {{ border-radius: 20px !important; overflow: hidden; }}
    .logo-circulaire {{ width: 150px; height: 150px; border-radius: 50%; object-fit: cover; animation: girar 8s linear infinite; }}
    @keyframes girar {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    [data-testid="stTextInput"] input {{ background-color: rgba(0,0,0,0) !important; border: 1px solid rgba(255,255,255,0.55) !important; color: white !important; width: 100% !important; }}
    [data-testid="stTextInput"] > div {{ background-color: rgba(0,0,0,0) !important; border: none !important; }}
    [data-testid="stTextInput"] > div > div {{ background-color: rgba(0,0,0,0) !important; }}
    [data-testid="stTextInput"] input::placeholder {{ color: rgba(255,255,255,0.55) !important; }}
    [data-testid="stSelectbox"] > div > div {{ background-color: rgba(0,0,0,0) !important; border: 1px solid rgba(255,255,255,0.55) !important; color: white !important; }}
    [data-testid="stSelectbox"] span {{ color: white !important; }}
    [data-testid="stSelectbox"] {{ cursor: pointer !important; }}
    [data-testid="stSelectbox"] * {{ cursor: pointer !important; }}
    [data-testid="stSidebar"] {{ background-color: rgba(0,0,0,0) !important; background: none !important; box-shadow: none !important; border-right: none !important; min-width: 80px !important; max-width: 80px !important; backdrop-filter: blur(4px); }}
    section[data-testid="stSidebar"] > div {{ background-color: rgba(0,0,0,0) !important; background: none !important; }}
    [data-testid="stSidebar"] button {{ background-color: transparent !important; border: none !important; width: 100% !important; padding: 25px 0 !important; height: 70px !important; }}
    [data-testid="stSidebar"] button:hover {{ background-color: rgba(255,255,255,0.1) !important; border-radius: 10px !important; }}
    [data-testid="stSidebar"] button p {{ display: none !important; }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    def icono_b64(nom):
        with open(nom, "rb") as f:
            return base64.b64encode(f.read()).decode()

    pages = [
        ('accueil', 'Accueil.png', 'Accueil'),
        ('favoris', 'Favoris.png', 'Favoris'),
        ('notifications', 'Notifications.png', 'Notifications'),
        ('profil', 'Profil.png', 'Profil'),
        ('kpi', 'KPI.png', 'KPI')
    ]

    for page, fichier, label in pages:
        img = icono_b64(fichier)
        clicked = st.button("​", key=f"btn_{page}", use_container_width=True, help=label)
        if clicked:
            st.session_state['page'] = page
            if page == 'accueil':
                for k in [k for k in st.session_state.keys() if k.startswith('show_reco_')]:
                    del st.session_state[k]
                for k in ['film_aleatoire', 'film_france', 'film_classique', 'film_action2']:
                    st.session_state[k] = None
                st.session_state['films_aleatoires'] = df_films.dropna(
                    subset=['poster_path']).sample(10).reset_index(drop=True)
                st.session_state['cat_france'] = df_france.sample(
                    min(10, len(df_france))).reset_index(drop=True)
                st.session_state['cat_classique'] = df_classique.sample(
                    min(10, len(df_classique))).reset_index(drop=True)
                st.session_state['cat_action2'] = df_action.sample(
                    min(10, len(df_action))).reset_index(drop=True)
            st.rerun()
        st.markdown(f'''
            <div style="text-align:center;margin-top:-62px;margin-bottom:15px;pointer-events:none;">
                <img src="data:image/png;base64,{img}" style="width:38px;">
            </div>''', unsafe_allow_html=True)

col_logo, col_titre = st.columns([1, 8])
with col_logo:
    st.markdown(f'<img src="data:image/png;base64,{logo_base64}" class="logo-circulaire">', unsafe_allow_html=True)
with col_titre:
    st.markdown("""
        <h1 style="font-size:90px;font-weight:900;letter-spacing:6px;
        background:linear-gradient(45deg,#c0392b,#e74c3c,#e67e22,#f39c12,#f1c40f);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin:0;">CINÉCREUSE</h1>""", unsafe_allow_html=True)

if st.session_state['page'] == 'favoris':
    st.markdown("### ❤️ Mes Favoris")
    if st.session_state['favoris']:
        films_favoris = df_films[df_films['tconst'].isin(st.session_state['favoris'])]
        for _, film in films_favoris.iterrows():
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                st.image(f"https://image.tmdb.org/t/p/w500{film['poster_path']}", width=150)
            with col2:
                st.markdown(f"### {get_titre(film)} ({int(film['startYear'])})")
                st.markdown(f"**Genres :** {film['genres']}")
                st.markdown(f"**Note IMDb :** ⭐ {film['averageRating']}")
            with col3:
                if st.button("🗑️ Retirer", key=f"retirer_{film['tconst']}"):
                    st.session_state['favoris'].remove(film['tconst'])
                    st.rerun()
            st.divider()
    else:
        st.info("Aucun favori. Ajoutez des films depuis l'accueil!")

elif st.session_state['page'] == 'notifications':
    st.markdown("### 🔔 Notifications")
    st.info("Pas de nouvelles notifications.")

elif st.session_state['page'] == 'profil':
    st.markdown("### 👤 Mon Profil")
    st.markdown("**Nom :** Utilisateur CinéCreuse")
    st.markdown(f"**Favoris :** {len(st.session_state['favoris'])} films")

elif st.session_state['page'] == 'kpi':
    st.markdown("""<h2 style="background:linear-gradient(45deg,#c0392b,#e74c3c,#e67e22,#f39c12,#f1c40f);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        📊 Tableau de bord — KPI</h2>""", unsafe_allow_html=True)

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("#### 🎭 Top 10 genres")
        genres_count = df_films['genres'].dropna().str.split(',').explode().str.strip()
        top_genres = genres_count.value_counts().head(10).reset_index()
        top_genres.columns = ['genre', 'nombre']
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        style_graph(fig1, ax1, 'Top 10 genres')
        colors = ['#c0392b','#d44e2a','#e67e22','#e8922a','#f0a030','#f39c12','#f5a623','#f7b733','#f9ca44','#f1c40f']
        sns.barplot(data=top_genres, x='nombre', y='genre', palette=colors, ax=ax1)
        ax1.set_xlabel('Nombre de films', color='white')
        ax1.set_ylabel('')
        st.pyplot(fig1)
        st.markdown("*Drama et Comedy dominent la production cinématographique mondiale.*")

    with col_k2:
        st.markdown("#### 📅 Évolution films par décennie")
        df_decade = df_films.copy()
        df_decade['decennie'] = (df_decade['startYear'] // 10 * 10).astype(int)
        decade_count = df_decade.groupby('decennie').size().reset_index(name='nombre')
        decade_count = decade_count[decade_count['decennie'] >= 1920]
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        style_graph(fig2, ax2, 'Évolution des films par décennie')
        ax2.plot(decade_count['decennie'], decade_count['nombre'], color='#e67e22', linewidth=2.5)
        ax2.fill_between(decade_count['decennie'], decade_count['nombre'], alpha=0.3, color='#f39c12')
        ax2.set_xlabel('Décennie', color='white')
        ax2.set_ylabel('Nombre de films', color='white')
        st.pyplot(fig2)
        st.markdown("*La production explose à partir des années 2000.*")

    st.divider()
    col_k3, col_k4 = st.columns(2)
    with col_k3:
        st.markdown("#### ⭐ Distribution des notes IMDb")
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        style_graph(fig5, ax5, 'Distribution des notes IMDb')
        ax5.hist(df_films['averageRating'].dropna(), bins=20, color='#e74c3c', edgecolor='#f39c12', linewidth=0.5)
        ax5.set_xlabel('Note IMDb', color='white')
        ax5.set_ylabel('Nombre de films', color='white')
        st.pyplot(fig5)
        st.markdown("*La majorité des films ont une note entre 6 et 7.5.*")

    with col_k4:
        st.markdown("#### 🏆 Top 10 films les mieux notés")
        top_films = df_films[df_films['numVotes'] >= 1000].sort_values('averageRating', ascending=False).head(10)
        fig6, ax6 = plt.subplots(figsize=(7, 5))
        style_graph(fig6, ax6, 'Top 10 films les mieux notés')
        colors2 = ['#c0392b','#d44e2a','#e67e22','#e8922a','#f0a030','#f39c12','#f5a623','#f7b733','#f9ca44','#f1c40f']
        sns.barplot(data=top_films, x='averageRating', y='titre', palette=colors2, ax=ax6)
        ax6.set_xlabel('Note IMDb', color='white')
        ax6.set_ylabel('')
        st.pyplot(fig6)
        st.markdown("*Les films classiques dominent le top 10.*")

    st.divider()
    col_k5, col_k6 = st.columns(2)
    with col_k5:
        st.markdown("#### 🌍 Top 10 pays de production")
        def parse_countries(x):
            try:
                return ast.literal_eval(x)
            except:
                return []
        pays_explode = df_films['production_countries'].dropna().apply(parse_countries).explode()
        top_pays = pays_explode.value_counts().head(10).reset_index()
        top_pays.columns = ['pays', 'nombre']
        fig7, ax7 = plt.subplots(figsize=(7, 5))
        style_graph(fig7, ax7, 'Top 10 pays de production')
        sns.barplot(data=top_pays, x='nombre', y='pays', palette=colors, ax=ax7)
        ax7.set_xlabel('Nombre de films', color='white')
        ax7.set_ylabel('')
        st.pyplot(fig7)
        st.markdown("*Les États-Unis dominent largement la production mondiale.*")

    with col_k6:
        st.markdown("#### 💰 Budget vs Recettes")
        df_budget = df_films[(df_films['budget'] > 0) & (df_films['revenue'] > 0)].copy()
        fig8, ax8 = plt.subplots(figsize=(7, 5))
        style_graph(fig8, ax8, 'Budget vs Recettes')
        ax8.scatter(df_budget['budget'], df_budget['revenue'], alpha=0.5, color='#e74c3c', s=20)
        ax8.set_xlabel('Budget ($)', color='white')
        ax8.set_ylabel('Recettes ($)', color='white')
        st.pyplot(fig8)
        st.markdown("*Les films à gros budget génèrent généralement plus de recettes.*")

else:
    st.markdown("<br>", unsafe_allow_html=True)
    col_search, col_genre = st.columns([4, 1])
    with col_search:
        film_input = st.text_input("", placeholder="🔍 Rechercher un film, un acteur...",
                                   label_visibility="collapsed")
    with col_genre:
        tous_genres = sorted(df_films['genres'].dropna().str.split(',')
                            .explode().str.strip().unique().tolist())
        tous_genres.insert(0, "🎭 Tous les genres")
        genre_selectionne = st.selectbox("", tous_genres, label_visibility="collapsed")

    df_filtre = df_films.copy()
    if genre_selectionne != "🎭 Tous les genres":
        df_filtre = df_filtre[df_filtre['genres'].str.contains(genre_selectionne, na=False)]

    if film_input:
        r_titre = df_filtre[
            df_filtre['titre'].str.contains(film_input, case=False, na=False) |
            df_filtre['primaryTitle'].str.contains(film_input, case=False, na=False)
        ]
        r_acteur = df_filtre[
            df_filtre['acteurs'].fillna('').str.contains(film_input, case=False, na=False)
        ] if 'acteurs' in df_filtre.columns else pd.DataFrame()
        r_genre = df_filtre[
            df_filtre['genres'].str.contains(film_input, case=False, na=False)
        ]
        resultats = pd.concat([r_titre, r_acteur, r_genre]).drop_duplicates(
            subset=['tconst']).reset_index(drop=True)

        if not resultats.empty:
            st.success(f"{len(resultats)} film(s) trouvé(s) pour **{film_input}**")
            st.markdown("### 🎬 Résultats de recherche")
            afficher_resultats(resultats.head(20), "recherche")
        else:
            st.warning("⚠️ Aucun résultat trouvé. Essayez un autre titre, acteur ou genre.")

    elif genre_selectionne != "🎭 Tous les genres":
        resultats = df_filtre.sort_values('averageRating', ascending=False).head(20)
        if not resultats.empty:
            st.markdown("### 🎬 Résultats")
            afficher_resultats(resultats, "filtre")

    afficher_categorie(
        "🤟 Films à découvrir aujourd'hui",
        st.session_state['films_aleatoires'],
        "aleatoire", "film_aleatoire"
    )
    afficher_categorie( 
        "👑 Films Français",
        st.session_state['cat_france'],
        "france", "film_france"
    )
    afficher_categorie(
        "🏛 Films Classiques",
        st.session_state['cat_classique'],
        "classique", "film_classique"
    )
    afficher_categorie(
        "🚀 Films Action",
        st.session_state['cat_action2'],
        "action2", "film_action2"
    )

st.divider()
st.markdown("<center>Wild Code School 2026 — Projet 2</center>", unsafe_allow_html=True)