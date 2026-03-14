# 🎬 Netflix Data Analytics Dashboard

An advanced, industry-level Streamlit application for analyzing Netflix content data. Perfect for Data Analyst internship portfolios.

## 🚀 Features

### 📊 Data Cleaning & Processing
- Handle missing values in country, director, and cast columns
- Convert date_added to datetime format
- Extract year_added and month_added columns
- Process duration data for numerical analysis

### 🎛️ Interactive Dashboard
- **Wide layout** with professional UI design
- **Sidebar filters** for dynamic data exploration:
  - Content Type (Movie/TV Show)
  - Country selection
  - Release Year range slider
  - Genre filtering

### 📈 Key Performance Indicators (KPIs)
- Total Titles count
- Movies vs TV Shows breakdown
- Most Common Genre
- Most Active Country (by content production)

### 📊 Advanced Visualizations
- **Movies vs TV Shows Distribution** (Pie Chart)
- **Content Release Trend** over time (Line Chart)
- **Top 10 Countries** producing Netflix content (Bar Chart)
- **Top 10 Genres** on Netflix (Bar Chart)
- **Duration Distribution** for movies (Histogram)
- **Ratings Distribution** (Bar Chart)
- **Content Addition Heatmap** by month and year

### 🔍 Search Functionality
- Real-time title search with instant results
- Filter-compatible search results

### 💡 Content Insights
- Most Common Director
- Most Frequent Actor
- Fastest Growing Country (based on recent additions)

### 🤖 Machine Learning Recommendation System
- **Genre-based similarity** using TF-IDF and Cosine Similarity
- User selects a title → Gets 5 similar recommendations
- Powered by scikit-learn

### ⚡ Performance Optimizations
- `@st.cache_data` for efficient data loading
- Modular code structure for maintainability

## 🛠️ Technology Stack

- **Python 3.x**
- **Streamlit** - Web app framework
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning for recommendations
- **NumPy** - Numerical computations

## 📁 Project Structure

```
netflix-data-project/
├── app.py                 # Main Streamlit application
├── netflix_titles.csv     # Netflix dataset
└── README.md             # Project documentation
```

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install streamlit pandas plotly scikit-learn
```

### Run the Application
```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## 📊 Dataset

The application uses the `netflix_titles.csv` dataset containing:
- 8,807 Netflix titles
- Information about movies and TV shows
- Metadata including cast, director, country, ratings, etc.

## 🎯 Key Highlights for Resume

- **Data Cleaning & Preprocessing**: Professional data handling techniques
- **Interactive Visualizations**: Multiple chart types with Plotly
- **Machine Learning Integration**: Recommendation system implementation
- **Performance Optimization**: Caching and efficient data processing
- **Professional UI/UX**: Clean, responsive design with emojis and organized sections
- **Smart Filtering**: Reset filters functionality and clear data status indicators
- **Improved Visualizations**: Clean area chart replacing complex heatmap

## 🐛 Issues Fixed

- **Visualization Function Call**: Fixed missing `create_visualizations()` call in main function
- **Area Chart Labels**: Corrected x/y axis labels for content additions chart
- **Error Handling**: All charts now have proper fallbacks for empty data
- **Layout Organization**: Improved chart arrangement with clear sections

## ✅ Current Status

- **All 7 Visualizations Working**: Pie, Line, Bar, Histogram, Area charts all functional
- **Interactive Filters**: Reset button and clear data status indicators
- **Clean UI**: Professional layout with proper spacing and organization
- **Error-Free**: No runtime errors or crashes
- **Improved UX**: Clear indication of active filters and data scope
- **Modular Code Structure**: Production-ready Python code

## 🔧 Code Quality & Error Handling

- **Robust Error Handling**: All visualizations include fallback displays for empty data
- **Data Validation**: Comprehensive checks for NaN values and invalid data
- **Exception Handling**: Try-catch blocks prevent crashes from data processing errors
- **Modular Functions**: Clean separation of concerns with dedicated functions
- **Performance Optimization**: Efficient data processing and caching
- **Cross-platform Compatibility**: Works on all operating systems

## 🐛 Bug Fixes Applied

- **Heatmap Visualization**: Fixed dimension mismatch between data and labels
- **Duration Processing**: Improved handling of missing duration data
- **Genre Filtering**: Enhanced regex-based filtering with case-insensitive matching
- **Recommendation System**: Added error handling for edge cases
- **Empty Data Handling**: All charts now display appropriate messages when no data is available
- **Month/Year Processing**: Proper filtering of invalid dates and months

## 📈 Sample Insights

The dashboard provides valuable insights such as:
- Netflix has more Movies than TV Shows
- United States is the most active content producer
- Documentaries and Dramas are the most common genres
- Content addition peaked in certain years
- Rating distributions and duration patterns

---

**Perfect for Data Analyst portfolios and internship applications!** 🎉</content>
<parameter name="filePath">/Users/vishwastiwari/netflix-data-project/README.md
