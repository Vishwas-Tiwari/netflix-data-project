import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("netflix_titles.csv")

st.title("Netflix Data Analytics Dashboard")

type_count = df['type'].value_counts()

fig = px.bar(x=type_count.index, y=type_count.values,
             labels={'x':'Type','y':'Count'},
             title="Movies vs TV Shows")

st.plotly_chart(fig)

year_data = df['release_year'].value_counts().sort_index()

fig2 = px.line(x=year_data.index, y=year_data.values,
               labels={'x':'Year','y':'Count'},
               title="Content Release Trend")

st.plotly_chart(fig2)

country_data = df['country'].value_counts().head(10)

fig3 = px.bar(x=country_data.index, y=country_data.values,
              title="Top Countries Producing Netflix Content")

st.plotly_chart(fig3)

selected_type = st.selectbox("Select Type", df['type'].unique())

filtered = df[df['type'] == selected_type]

st.write(filtered.head())