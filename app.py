import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import re

# Set page configuration
st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load the Netflix dataset"""
    df = pd.read_csv("netflix_titles.csv")
    return df

# Data cleaning function
def clean_data(df):
    """Clean and preprocess the data"""
    # Handle missing values
    df['country'] = df['country'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')

    # Convert date_added to datetime
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    # Extract year_added and month_added
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    # Fill NaN values in year_added and month_added
    df['year_added'] = df['year_added'].fillna(0).astype(int)
    df['month_added'] = df['month_added'].fillna(0).astype(int)

    # Clean duration - extract numeric values
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)

    return df

# Create KPI metrics
def create_kpis(df):
    """Create KPI metrics"""
    total_titles = len(df)
    total_movies = len(df[df['type'] == 'Movie'])
    total_tv_shows = len(df[df['type'] == 'TV Show'])

    # Most common genre
    all_genres = []
    for genres in df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])
    most_common_genre = Counter(all_genres).most_common(1)[0][0] if all_genres else 'N/A'

    # Most active country
    if not df.empty and 'country' in df.columns:
        country_counts = df['country'].value_counts()
        most_active_country = country_counts.index[0] if not country_counts.empty else 'N/A'
    else:
        most_active_country = 'N/A'

    return total_titles, total_movies, total_tv_shows, most_common_genre, most_active_country

# Create visualizations
def create_visualizations(df, filtered_df):
    """Create all visualizations"""

    # Custom plotly layout dictionary matching Netflix identity
    plotly_layout = {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Montserrat, sans-serif', 'color': '#f5f5f7'},
        'title': {'font': {'family': 'Montserrat, sans-serif', 'size': 18, 'color': '#ffffff'}},
        'xaxis': {
            'gridcolor': '#222222',
            'linecolor': '#333333',
            'zerolinecolor': '#333333',
            'tickfont': {'color': '#aaaaaa'}
        },
        'yaxis': {
            'gridcolor': '#222222',
            'linecolor': '#333333',
            'zerolinecolor': '#333333',
            'tickfont': {'color': '#aaaaaa'}
        },
        'margin': dict(l=40, r=40, t=60, b=40)
    }

    # Movies vs TV Shows
    type_count = filtered_df['type'].value_counts()
    if not type_count.empty:
        fig1 = px.pie(values=type_count.values, names=type_count.index,
                      title="🎬 Movies vs TV Shows Distribution",
                      color_discrete_sequence=['#E50914', '#221F1F'])
        fig1.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            marker=dict(line=dict(color='#111111', width=2))
        )
    else:
        fig1 = go.Figure()
        fig1.add_annotation(text="No data available", showarrow=False)
    fig1.update_layout(**plotly_layout)

    # Content release trend
    year_data = filtered_df['release_year'].value_counts().sort_index()
    if not year_data.empty:
        fig2 = px.line(x=year_data.index, y=year_data.values,
                       labels={'x':'Release Year','y':'Content Count'},
                       title="📈 Content Release Trend Over Time")
        fig2.update_traces(line_color='#E50914', line_width=3)
    else:
        fig2 = go.Figure()
        fig2.add_annotation(text="No data available", showarrow=False)
    fig2.update_layout(**plotly_layout)

    # Top 10 countries
    country_data = filtered_df['country'].value_counts().head(10)
    if not country_data.empty:
        fig3 = px.bar(x=country_data.index, y=country_data.values,
                      labels={'x': 'Country', 'y': 'Count'},
                      title="🌍 Top 10 Countries Producing Netflix Content",
                      color=country_data.values,
                      color_continuous_scale=['#221F1F', '#E50914'])
        fig3.update_layout(coloraxis_showscale=False)
        fig3.update_traces(marker_line_color='#111111', marker_line_width=1)
    else:
        fig3 = go.Figure()
        fig3.add_annotation(text="No data available", showarrow=False)
    fig3.update_layout(**plotly_layout)

    # Top 10 genres
    all_genres = []
    for genres in filtered_df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])
    if all_genres:
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        fig4 = px.bar(x=genre_counts.index, y=genre_counts.values,
                      labels={'x': 'Genre', 'y': 'Count'},
                      title="🎭 Top 10 Genres on Netflix",
                      color=genre_counts.values,
                      color_continuous_scale=['#221F1F', '#E50914'])
        fig4.update_layout(coloraxis_showscale=False)
        fig4.update_traces(marker_line_color='#111111', marker_line_width=1)
    else:
        fig4 = go.Figure()
        fig4.add_annotation(text="No data available", showarrow=False)
    fig4.update_layout(**plotly_layout)

    # Duration distribution (works for both Movies and TV Shows)
    if 'duration_num' in filtered_df.columns:
        duration_data = filtered_df['duration_num'].dropna()
        if not duration_data.empty:
            duration_label = "Duration (minutes/seasons)"
            fig5 = px.histogram(x=duration_data.values,
                               title="⏱️ Duration Distribution",
                               labels={'x': duration_label, 'y': 'Count'},
                               nbins=30,
                               color_discrete_sequence=['#E50914'])
            fig5.update_traces(marker=dict(line=dict(color='#111111', width=1)))
        else:
            fig5 = go.Figure()
            fig5.add_annotation(text="No duration data available", showarrow=False)
    else:
        fig5 = go.Figure()
        fig5.add_annotation(text="No duration data available", showarrow=False)
    fig5.update_layout(**plotly_layout)

    # Ratings distribution
    rating_data = filtered_df['rating'].value_counts()
    if not rating_data.empty:
        fig6 = px.bar(x=rating_data.index, y=rating_data.values,
                      labels={'x': 'Rating', 'y': 'Count'},
                      title="⭐ Ratings Distribution",
                      color=rating_data.values,
                      color_continuous_scale=['#221F1F', '#E50914'])
        fig6.update_layout(coloraxis_showscale=False)
        fig6.update_traces(marker_line_color='#111111', marker_line_width=1)
    else:
        fig6 = go.Figure()
        fig6.add_annotation(text="No rating data available", showarrow=False)
    fig6.update_layout(**plotly_layout)

    # Content additions over time
    content_additions = filtered_df[filtered_df['year_added'] > 0]['year_added'].value_counts().sort_index()
    if not content_additions.empty:
        fig7 = px.area(x=content_additions.index, y=content_additions.values,
                      title="📅 Content Additions by Year",
                      labels={'x':'Year Added', 'y':'Count of Titles'},
                      color_discrete_sequence=['#E50914'])
        fig7.update_traces(mode='lines+markers', marker=dict(size=6, color='#ffffff', line=dict(color='#E50914', width=2)))
    else:
        fig7 = go.Figure()
        fig7.add_annotation(text="No content addition data available", showarrow=False)
    fig7.update_layout(**plotly_layout)

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# Recommendation system
def get_recommendations(df, title, top_n=5):
    """Get content recommendations based on genre similarity"""
    if df.empty or title not in df['title'].values:
        return pd.DataFrame()

    try:
        # Reset index to map integer position to rows consistently
        df_reset = df.reset_index(drop=True)
        
        # Prepare TF-IDF on genres
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df_reset['listed_in'].fillna(''))

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get index of the title
        idx = df_reset[df_reset['title'] == title].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:min(len(sim_scores), top_n+1)]  # Exclude itself

        # Get recommended titles
        title_indices = [i[0] for i in sim_scores]
        recommendations = df_reset.iloc[title_indices][['title', 'type', 'listed_in', 'rating']]

        return recommendations
    except Exception as e:
        print(f"Recommendation error: {e}")
        return pd.DataFrame()

# Card HTML helper for KPI & Insights
def card_html(title, value, icon):
    return f"""
    <div style="
        background-color: #181818;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-bottom: 15px;
    " onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='#E50914'; this.style.boxShadow='0 8px 20px rgba(229, 9, 20, 0.2)';" onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='#333333'; this.style.boxShadow='0 4px 12px rgba(0, 0, 0, 0.4)';">
        <div style="font-size: 32px; margin-bottom: 8px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));">{icon}</div>
        <div style="font-size: 11px; color: #aaaaaa; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px;">{title}</div>
        <div title="{value}" style="font-size: 16px; font-weight: 800; color: #ffffff; width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{value}</div>
    </div>
    """

# Main app
def main():
    # Inject custom CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;800&display=swap');
        
        /* Apply Montserrat to all text */
        html, body, [class*="css"], .stApp {
            font-family: 'Montserrat', sans-serif !important;
        }
        
        /* Background and primary colors */
        .stApp {
            background-color: #111111;
            color: #f5f5f7;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #141414 !important;
            border-right: 1px solid #2b2b2b !important;
        }
        
        /* Custom header/title styles */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 700 !important;
        }
        
        /* Main page title styling */
        .dashboard-title {
            color: #E50914 !important;
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            text-shadow: 0 0 15px rgba(229, 9, 20, 0.4);
            margin-bottom: 0.2rem;
            text-transform: uppercase;
            letter-spacing: -1px;
        }
        
        .dashboard-subtitle {
            color: #aaaaaa;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 1.5rem;
        }
        
        /* Netflix Red gradient divider */
        .netflix-divider {
            height: 4px;
            background: linear-gradient(90deg, #E50914 0%, #221F1F 50%, #111111 100%);
            margin-bottom: 2rem;
            border-radius: 2px;
        }
        
        /* Form inputs styling */
        .stTextInput>div>div>input {
            background-color: #181818 !important;
            color: #ffffff !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
            padding: 10px 14px !important;
        }
        
        .stSelectbox>div>div>div {
            background-color: #181818 !important;
            color: #ffffff !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
        }
        
        /* Multiselect styling */
        div[data-baseweb="select"] {
            background-color: #181818 !important;
            border: 1px solid #333333 !important;
            border-radius: 6px !important;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #E50914 !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 12px 24px !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            transition: all 0.2s ease-in-out !important;
            width: 100%;
        }
        
        .stButton>button:hover {
            background-color: #f40b16 !important;
            box-shadow: 0 0 10px rgba(229, 9, 20, 0.6);
            transform: scale(1.02);
        }
        
        .stButton>button:active {
            transform: scale(0.98);
        }
        
        /* Table / DataFrame container styling */
        .stDataFrame {
            background-color: #181818 !important;
            border: 1px solid #333333 !important;
            border-radius: 8px !important;
        }
        
        /* Success/Info alerts */
        .stAlert {
            background-color: #181818 !important;
            color: #ffffff !important;
            border: 1px solid #E50914 !important;
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title block
    st.markdown('<div class="dashboard-title">🎬 Netflix Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">An interactive, industry-level content analysis dashboard powered by Streamlit and Plotly.</div>', unsafe_allow_html=True)
    st.markdown('<div class="netflix-divider"></div>', unsafe_allow_html=True)

    # Load and clean data
    df = load_data()
    df = clean_data(df)

    # Sidebar filters
    st.sidebar.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem;">🎛️ Dashboard Filters</div>', unsafe_allow_html=True)

    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()

    st.sidebar.markdown("---")

    # Type filter
    type_options = df['type'].unique()
    selected_types = st.sidebar.multiselect("Select Content Type", type_options, default=type_options)

    # Country filter
    country_options = sorted(df['country'].unique())
    selected_countries = st.sidebar.multiselect("Select Producing Country", country_options, default=[])

    # Release year range
    min_year = int(df['release_year'].min())
    max_year = int(df['release_year'].max())
    year_range = st.sidebar.slider("Release Year Range", min_year, max_year, (min_year, max_year))

    # Genre filter
    all_genres = set()
    for genres in df['listed_in'].dropna():
        all_genres.update([g.strip() for g in genres.split(',')])
    genre_options = sorted(list(all_genres))
    selected_genres = st.sidebar.multiselect("Select Genre", genre_options, default=[])

    # Apply filters
    filtered_df = df[df['type'].isin(selected_types)]
    if selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    filtered_df = filtered_df[(filtered_df['release_year'] >= year_range[0]) &
                             (filtered_df['release_year'] <= year_range[1])]
    if selected_genres:
        # More robust genre filtering
        genre_filter = filtered_df['listed_in'].str.contains('|'.join(selected_genres), case=False, na=False)
        filtered_df = filtered_df[genre_filter]

    # Show current filter status
    active_filters = []
    if len(selected_types) < len(type_options):
        active_filters.append(f"Type: {', '.join(selected_types)}")
    if selected_countries:
        active_filters.append(f"Countries: {len(selected_countries)} selected")
    if year_range != (min_year, max_year):
        active_filters.append(f"Years: {year_range[0]}-{year_range[1]}")
    if selected_genres:
        active_filters.append(f"Genres: {len(selected_genres)} selected")

    if active_filters:
        st.info(f"📊 **Filtered Catalog** - Showing {len(filtered_df):,} of {len(df):,} titles | Active filters: {', '.join(active_filters)}")
    else:
        st.success(f"📊 **Complete Catalog** - Showing {len(filtered_df):,} titles | Complete Netflix catalog (Movies & TV Shows)")

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Section
    st.header("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_titles, total_movies, total_tv_shows, most_common_genre, most_active_country = create_kpis(filtered_df)

    with col1:
        st.markdown(card_html("Total Titles", f"{total_titles:,}", "🎬"), unsafe_allow_html=True)
    with col2:
        st.markdown(card_html("Total Movies", f"{total_movies:,}", "🎥"), unsafe_allow_html=True)
    with col3:
        st.markdown(card_html("Total TV Shows", f"{total_tv_shows:,}", "📺"), unsafe_allow_html=True)
    with col4:
        st.markdown(card_html("Most Common Genre", most_common_genre, "🎭"), unsafe_allow_html=True)
    with col5:
        st.markdown(card_html("Most Active Country", most_active_country, "🌍"), unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Visualizations
    st.header("📈 Advanced Visualizations")

    fig1, fig2, fig3, fig4, fig5, fig6, fig7 = create_visualizations(df, filtered_df)

    # Row 1: Overview charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True, key="fig_type_pie")
    with col2:
        st.plotly_chart(fig2, use_container_width=True, key="fig_release_trend")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Geographic and Genre analysis
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig3, use_container_width=True, key="fig_top_countries")
    with col4:
        st.plotly_chart(fig4, use_container_width=True, key="fig_top_genres")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Content details
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(fig5, use_container_width=True, key="fig_duration_dist")
    with col6:
        st.plotly_chart(fig6, use_container_width=True, key="fig_ratings_dist")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 4: Content growth
    st.plotly_chart(fig7, use_container_width=True, key="fig_additions_area")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Search Feature
    st.header("🔍 Search Titles")
    search_term = st.text_input("Enter search keywords for Netflix titles (e.g. Stranger Things, Narcos):")
    if search_term:
        search_results = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
        if not search_results.empty:
            st.dataframe(search_results[['title', 'type', 'country', 'release_year', 'rating']], use_container_width=True)
        else:
            st.info("No Netflix titles match your search criteria. Try a different title!")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Top Content Insights
    st.header("💡 Top Content Insights")

    col1, col2, col3 = st.columns(3)

    # Most common director
    director_counts = filtered_df['director'].value_counts()
    if len(director_counts) > 1:
        most_common_director = director_counts.index[1]  # Skip 'Unknown'
    else:
        most_common_director = 'N/A'

    # Most frequent actor
    all_cast = []
    for cast in filtered_df['cast'].dropna():
        if cast != 'Unknown':
            all_cast.extend([c.strip() for c in cast.split(',')])
    if all_cast:
        most_frequent_actor = Counter(all_cast).most_common(1)[0][0]
    else:
        most_frequent_actor = 'N/A'

    # Country with fastest growing content (simplified: highest count in recent years)
    recent_years = filtered_df[filtered_df['year_added'] >= 2020]
    if not recent_years.empty:
        fastest_growing_country = recent_years['country'].value_counts().index[0]
    else:
        fastest_growing_country = 'N/A'

    with col1:
        st.markdown(card_html("Most Common Director", most_common_director, "🎬"), unsafe_allow_html=True)
    with col2:
        st.markdown(card_html("Most Frequent Actor", most_frequent_actor, "👤"), unsafe_allow_html=True)
    with col3:
        st.markdown(card_html("Fastest Growing Country", fastest_growing_country, "📈"), unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Recommendation System
    st.header("🤖 Content Recommendation System")
    st.write("Select a title to get genre-based similarity recommendations powered by TF-IDF & Cosine Similarity:")

    title_options = filtered_df['title'].tolist()
    if title_options:
        selected_title = st.selectbox("Choose a title:", title_options)
        if selected_title:
            recommendations = get_recommendations(filtered_df, selected_title)
            if not recommendations.empty:
                st.write("**Top 5 Recommended Titles:**")
                st.dataframe(recommendations, use_container_width=True)
            else:
                st.info("No recommendations found for this title.")
    else:
        st.info("No titles match your current filters. Adjust your filters in the sidebar to explore and search titles.")

if __name__ == "__main__":
    main()