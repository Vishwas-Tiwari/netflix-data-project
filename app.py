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
    most_active_country = df['country'].value_counts().index[0] if not df['country'].empty else 'N/A'

    return total_titles, total_movies, total_tv_shows, most_common_genre, most_active_country

# Create visualizations
def create_visualizations(df, filtered_df):
    """Create all visualizations"""

    # Movies vs TV Shows
    type_count = filtered_df['type'].value_counts()
    if not type_count.empty:
        fig1 = px.pie(values=type_count.values, names=type_count.index,
                      title="🎬 Movies vs TV Shows Distribution",
                      color_discrete_sequence=px.colors.qualitative.Set3)
    else:
        fig1 = go.Figure()
        fig1.add_annotation(text="No data available", showarrow=False)

    # Content release trend
    year_data = filtered_df['release_year'].value_counts().sort_index()
    if not year_data.empty:
        fig2 = px.line(x=year_data.index, y=year_data.values,
                       labels={'x':'Year','y':'Count'},
                       title="📈 Content Release Trend Over Time")
    else:
        fig2 = go.Figure()
        fig2.add_annotation(text="No data available", showarrow=False)

    # Top 10 countries
    country_data = filtered_df['country'].value_counts().head(10)
    if not country_data.empty:
        fig3 = px.bar(x=country_data.index, y=country_data.values,
                      title="🌍 Top 10 Countries Producing Netflix Content",
                      color=country_data.values,
                      color_continuous_scale='Blues')
    else:
        fig3 = go.Figure()
        fig3.add_annotation(text="No data available", showarrow=False)

    # Top 10 genres
    all_genres = []
    for genres in filtered_df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])
    if all_genres:
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        fig4 = px.bar(x=genre_counts.index, y=genre_counts.values,
                      title="🎭 Top 10 Genres on Netflix",
                      color=genre_counts.values,
                      color_continuous_scale='Greens')
    else:
        fig4 = go.Figure()
        fig4.add_annotation(text="No data available", showarrow=False)

    # Duration distribution (works for both Movies and TV Shows)
    if 'duration_num' in filtered_df.columns:
        duration_data = filtered_df['duration_num'].dropna()
        if not duration_data.empty:
            # Label adjusts based on selected type(s)
            duration_label = "Duration (minutes/seasons)"
            fig5 = px.histogram(x=duration_data.values,
                               title="⏱️ Duration Distribution",
                               labels={'x': duration_label},
                               nbins=30)
        else:
            fig5 = go.Figure()
            fig5.add_annotation(text="No duration data available", showarrow=False)
    else:
        fig5 = go.Figure()
        fig5.add_annotation(text="No duration data available", showarrow=False)

    # Ratings distribution
    rating_data = filtered_df['rating'].value_counts()
    if not rating_data.empty:
        fig6 = px.bar(x=rating_data.index, y=rating_data.values,
                      title="⭐ Ratings Distribution",
                      color=rating_data.values,
                      color_continuous_scale='Reds')
    else:
        fig6 = go.Figure()
        fig6.add_annotation(text="No rating data available", showarrow=False)

    # Content additions over time (replacing messy heatmap)
    content_additions = filtered_df[filtered_df['year_added'] > 0]['year_added'].value_counts().sort_index()
    if not content_additions.empty:
        fig7 = px.area(x=content_additions.index, y=content_additions.values,
                      title="📅 Content Additions by Year",
                      labels={'x':'Year', 'y':'Content Added'},
                      color_discrete_sequence=['#FF6B6B'])
        fig7.update_traces(mode='lines+markers')
    else:
        fig7 = go.Figure()
        fig7.add_annotation(text="No content addition data available", showarrow=False)

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# Recommendation system
def get_recommendations(df, title, top_n=5):
    """Get content recommendations based on genre similarity"""
    if title not in df['title'].values or df.empty:
        return pd.DataFrame()

    try:
        # Prepare TF-IDF on genres
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df['listed_in'].fillna(''))

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get index of the title
        idx = df[df['title'] == title].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Exclude itself

        # Get recommended titles
        title_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[title_indices][['title', 'type', 'listed_in', 'rating']]

        return recommendations
    except Exception as e:
        print(f"Recommendation error: {e}")
        return pd.DataFrame()

# Main app
def main():
    st.title("🎬 Netflix Data Analytics Dashboard")
    st.markdown("---")

    # Load and clean data
    df = load_data()
    df = clean_data(df)

    # Sidebar filters
    st.sidebar.header("🎛️ Filters")

    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters", type="primary"):
        st.rerun()

    st.sidebar.markdown("---")

    # Type filter
    type_options = df['type'].unique()
    selected_types = st.sidebar.multiselect("Select Type", type_options, default=type_options)

    # Country filter
    country_options = sorted(df['country'].unique())
    selected_countries = st.sidebar.multiselect("Select Country", country_options, default=[])

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
        st.info(f"📊 **Filtered Data** - {len(filtered_df)} titles | Active filters: {', '.join(active_filters)}")
    else:
        st.success(f"📊 **All Data** - {len(filtered_df)} titles | Showing complete Netflix catalog (Movies & TV Shows)")

    st.markdown("---")

    # KPI Section
    st.header("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_titles, total_movies, total_tv_shows, most_common_genre, most_active_country = create_kpis(filtered_df)

    with col1:
        st.metric("Total Titles", total_titles)
    with col2:
        st.metric("Total Movies", total_movies)
    with col3:
        st.metric("Total TV Shows", total_tv_shows)
    with col4:
        st.metric("Most Common Genre", most_common_genre)
    with col5:
        st.metric("Most Active Country", most_active_country)

    st.markdown("---")

    # Visualizations
    st.header("📈 Advanced Visualizations")

    fig1, fig2, fig3, fig4, fig5, fig6, fig7 = create_visualizations(df, filtered_df)

    # Row 1: Overview charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, width='stretch')
    with col2:
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")

    # Row 2: Geographic and Genre analysis
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig3, width='stretch')
    with col4:
        st.plotly_chart(fig4, width='stretch')

    st.markdown("---")

    # Row 3: Content details
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(fig5, width='stretch')
    with col6:
        st.plotly_chart(fig6, width='stretch')

    st.markdown("---")

    # Row 4: Content growth
    st.plotly_chart(fig7, width='stretch')

    st.markdown("---")

    # Search Feature
    st.header("🔍 Search Titles")
    search_term = st.text_input("Search for a title:")
    if search_term:
        search_results = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
        if not search_results.empty:
            st.dataframe(search_results[['title', 'type', 'country', 'release_year', 'rating']].head(10))
        else:
            st.write("No titles found matching your search.")

    st.markdown("---")

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
        st.metric("Most Common Director", most_common_director)
    with col2:
        st.metric("Most Frequent Actor", most_frequent_actor)
    with col3:
        st.metric("Fastest Growing Country", fastest_growing_country)

    st.markdown("---")

    # Recommendation System
    st.header("🤖 Content Recommendation System")
    st.write("Select a title to get genre-based recommendations:")

    title_options = filtered_df['title'].tolist()
    selected_title = st.selectbox("Choose a title:", title_options)

    if selected_title:
        recommendations = get_recommendations(filtered_df, selected_title)
        if not recommendations.empty:
            st.write("Recommended titles:")
            st.dataframe(recommendations)
        else:
            st.write("No recommendations available.")

if __name__ == "__main__":
    main()