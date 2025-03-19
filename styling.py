import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Import the EVRecommendationSystem class
class EVRecommendationSystem:
    def __init__(self, data_path):
        """Initialize the recommendation system with the dataset."""
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """Preprocess the data for recommendation."""
        # Clean the data - replace '-' with NaN and then fill with 0
        self.df = self.df.replace('-', np.nan)
        
        # Convert columns to numeric
        numeric_columns = ['AccelSec', 'TopSpeed_KmH', 'Range_Km', 
                           'Efficiency_WhKm', 'FastCharge_KmH', 'PriceEuro']
        
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Fill missing values with 0 or appropriate values
        self.df['FastCharge_KmH'] = self.df['FastCharge_KmH'].fillna(0)
        self.df = self.df.fillna(0)
        
        # Create a full model name column for better display
        self.df['FullModelName'] = self.df['Brand'].str.strip() + ' ' + self.df['Model'].str.strip()
        
        # Create feature matrix for similarity calculation
        self.features = self.df[['AccelSec', 'TopSpeed_KmH', 'Range_Km', 
                                'Efficiency_WhKm', 'FastCharge_KmH', 'PriceEuro']]
        
        # Normalize features
        self.scaler = MinMaxScaler()
        self.features_normalized = self.scaler.fit_transform(self.features)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.features_normalized)
    
    def get_recommendations_by_model(self, model_name, top_n=5):
        """Get recommendations based on a specific model."""
        # Find the index of the model
        model_indices = self.df.index[self.df['FullModelName'].str.contains(model_name, case=False)].tolist()
        
        if not model_indices:
            return "Model not found. Please check the spelling."
        
        model_idx = model_indices[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[model_idx]))
        
        # Sort by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar models (excluding the input model)
        similar_models = similarity_scores[1:top_n+1]
        
        # Return the recommended models
        recommended_models = []
        for i, score in similar_models:
            recommended_models.append({
                'Model': self.df.iloc[i]['FullModelName'],
                'Similarity Score': f"{score:.2f}",
                'Acceleration (s)': self.df.iloc[i]['AccelSec'],
                'Top Speed (km/h)': self.df.iloc[i]['TopSpeed_KmH'],
                'Range (km)': self.df.iloc[i]['Range_Km'],
                'Price (€)': self.df.iloc[i]['PriceEuro']
            })
        
        return recommended_models
    
    def get_recommendations_by_preferences(self, preferences, top_n=5):
        """Get recommendations based on user preferences."""
        filtered_df = self.df.copy()
        
        # Filter by range
        if 'min_range' in preferences:
            filtered_df = filtered_df[filtered_df['Range_Km'] >= preferences['min_range']]
        
        # Filter by price
        if 'max_price' in preferences:
            filtered_df = filtered_df[filtered_df['PriceEuro'] <= preferences['max_price']]
        
        # Filter by body style
        if 'body_style' in preferences and preferences['body_style']:
            filtered_df = filtered_df[filtered_df['BodyStyle'] == preferences['body_style']]
        
        # Filter by seats
        if 'min_seats' in preferences:
            filtered_df = filtered_df[filtered_df['Seats'] >= preferences['min_seats']]
        
        # Filter by fast charge
        if 'fast_charge' in preferences and preferences['fast_charge']:
            filtered_df = filtered_df[filtered_df['RapidCharge'] == 'Yes']
        
        # Sort by range and efficiency
        filtered_df = filtered_df.sort_values(by=['Range_Km', 'Efficiency_WhKm'], ascending=[False, True])
        
        # Return top N models
        recommended_models = []
        for i in range(min(top_n, len(filtered_df))):
            car = filtered_df.iloc[i]
            recommended_models.append({
                'Model': car['FullModelName'],
                'Body Style': car['BodyStyle'],
                'Range (km)': car['Range_Km'],
                'Acceleration (s)': car['AccelSec'],
                'Efficiency (Wh/km)': car['Efficiency_WhKm'],
                'Fast Charge': car['RapidCharge'],
                'Price (€)': car['PriceEuro']
            })
        
        return recommended_models
    
    def get_available_body_styles(self):
        """Get all available body styles in the dataset."""
        return sorted(self.df['BodyStyle'].unique().tolist())
    
    def get_available_brands(self):
        """Get all available brands in the dataset."""
        return sorted(self.df['Brand'].str.strip().unique().tolist())
    
    def get_price_range(self):
        """Get the price range in the dataset."""
        return {
            'min': int(self.df['PriceEuro'].min()),
            'max': int(self.df['PriceEuro'].max())
        }
    
    def get_range_stats(self):
        """Get statistics about the range of EVs in the dataset."""
        return {
            'min': int(self.df['Range_Km'].min()),
            'max': int(self.df['Range_Km'].max()),
            'mean': int(self.df['Range_Km'].mean()),
            'median': int(self.df['Range_Km'].median())
        }

# Initialize the recommendation system
@st.cache_resource
def load_recommender():
    return EVRecommendationSystem('ElectricCarData_Clean.csv')

# Page configuration
st.set_page_config(page_title="BikeSetu - Best Electric Scooter Deals", layout="wide")

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Navigation functions
def go_to_recommendations():
    st.session_state.page = 'recommendations'

def go_to_chat():
    st.session_state.page = 'chat'

def go_to_home():
    st.session_state.page = 'home'

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: black !important;
        }
        .main-container {
            background-color: black;
            padding: 5% 10%;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 20px 40px;
            color: black;
            border-bottom: 1px solid #ddd;
        }
        .logo {
            font-weight: bold;
            font-size: 1.7rem;
        }
        .nav {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .location-btn {
            background-color: black;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 14px;
            border: none;
            cursor: pointer;
        }
        .main-text h1 {
            font-size: 3.5rem;
            line-height: 1.3;
            font-weight: 700;
        }
        .btn-container {
            margin-top: 30px;
            display: flex;
            gap: 15px;
        }
        .track-btn, .recommend-btn, .chat-btn, .back-btn {
            padding: 14px 28px;
            font-size: 16px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: 0.3s ease;
        }
        .track-btn {
            background-color: #222;
            color: white;
        }
        .track-btn:hover {
            background-color: #444;
        }
        .recommend-btn, .chat-btn {
            background-color: white;
            color: black;
        }
        .recommend-btn:hover, .chat-btn:hover {
            background-color: #f1f1f1;
        }
        .chat-btn {
            background-color: #4CAF50;
            color: white;
        }
        .chat-btn:hover {
            background-color: #45a049;
        }
        .back-btn {
            background-color: #f44336;
            color: white;
            margin-bottom: 20px;
        }
        .back-btn:hover {
            background-color: #d32f2f;
        }
        .scooter-img-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: black;
        }
        .scooter-img-container img {
            width: 350px;
            max-width: 100%;
            border-radius: 50%;
            box-shadow: 0 0 30px rgba(255, 136, 0, 0.3);
        }
        .recommendation-section {
            background-color: #f5f5f5;
            padding: 30px;
            border-radius: 15px;
            margin-top: 20px;
            color: black;
        }
        .chat-section {
            background-color: #f5f5f5;
            padding: 30px;
            border-radius: 15px;
            margin: 20px auto;
            max-width: 800px;
            color: black;
            text-align: center;
        }
        .chat-container {
            margin: 0 auto;
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Navigation")
st.sidebar.button("Home", on_click=go_to_home)
st.sidebar.button("AI Recommendations", on_click=go_to_recommendations)
st.sidebar.button("Chat with Us", on_click=go_to_chat)

# Header (shown on all pages)
st.markdown("""
    <div class="header">
        <div class="logo">BIKESETU</div>
        <div class="nav">
            <button class="location-btn">Location</button>
            <span>Whatsapp</span>
            <span style="font-size: 24px; cursor: pointer;">&#9776;</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# HOME PAGE
if st.session_state.page == 'home':
    # Main content container
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
            <div class="main-text">
                <h1>BEST DEALS<br>ON ELECTRIC SCOOTERS</h1>
                <div class="btn-container">
                    <button class="track-btn" id="track_btn">Track</button>
                    <button class="recommend-btn" id="recommend_btn">AI-Recommendation</button>
                    <button class="chat-btn" id="chat_btn">Chat with Us</button>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Button click handlers in Python
        if st.button("AI-Recommendation", key="ai_rec_btn"):
            st.session_state.page = 'recommendations'
            st.experimental_rerun()
            
        if st.button("Chat with Us", key="main_chat_btn"):
            st.session_state.page = 'chat'
            st.experimental_rerun()

    with col2:
        st.markdown("<div class='scooter-img-container'>", unsafe_allow_html=True)
        st.image("scooter2.png", use_column_width=False, width=350)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# RECOMMENDATIONS PAGE
elif st.session_state.page == 'recommendations':
    st.markdown("<div style='padding: 20px;'>", unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Home", key="back_from_rec"):
        st.session_state.page = 'home'
        st.experimental_rerun()
    
    st.markdown("<div class='recommendation-section'>", unsafe_allow_html=True)
    st.header("Electric Vehicle Recommendations")
    
    try:
        # Load the recommendation system
        recommender = load_recommender()
        
        # Create tabs for different recommendation methods
        tab1, tab2 = st.tabs(["By Preferences", "By Similar Model"])
        
        with tab1:
            st.subheader("Find Electric Vehicles Based on Your Preferences")
            
            # Get price range and range stats
            price_range = recommender.get_price_range()
            range_stats = recommender.get_range_stats()
            
            # User preferences
            col1, col2 = st.columns(2)
            
            with col1:
                min_range = st.slider("Minimum Range (km)", 
                                    int(range_stats['min']), 
                                    int(range_stats['max']), 
                                    int(range_stats['median']))
                
                max_price = st.slider("Maximum Price (€)", 
                                    int(price_range['min']), 
                                    int(price_range['max']), 
                                    int(price_range['max'] * 0.7))
            
            with col2:
                body_styles = ["Any"] + recommender.get_available_body_styles()
                body_style = st.selectbox("Body Style", body_styles)
                
                # Convert "Any" to None for the filter
                if body_style == "Any":
                    body_style = None
                
                min_seats = st.slider("Minimum Number of Seats", 2, 7, 4)
                fast_charge = st.checkbox("Fast Charging Required", value=True)
            
            # Number of recommendations
            num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
            
            if st.button("Find EVs"):
                preferences = {
                    'min_range': min_range,
                    'max_price': max_price,
                    'body_style': body_style,
                    'min_seats': min_seats,
                    'fast_charge': fast_charge
                }
                
                recommendations = recommender.get_recommendations_by_preferences(preferences, num_recommendations)
                
                if not recommendations:
                    st.warning("No vehicles match your criteria. Try adjusting your preferences.")
                else:
                    st.success(f"Found {len(recommendations)} matching vehicles")
                    
                    # Convert to DataFrame for better display
                    df_recommendations = pd.DataFrame(recommendations)
                    st.dataframe(df_recommendations)
        
        with tab2:
            st.subheader("Find Similar Electric Vehicles")
            st.write("Get recommendations based on a model you already like.")
            
            # Get available brands
            brands = recommender.get_available_brands()
            selected_brand = st.selectbox("Select Brand", brands)
            
            # Get models for selected brand
            brand_models = recommender.df[recommender.df['Brand'].str.strip() == selected_brand]['Model'].str.strip().unique().tolist()
            selected_model = st.selectbox("Select Model", brand_models)
            
            # Number of recommendations
            num_recommendations = st.slider("Number of similar models", 1, 10, 5, key="similar_models_slider")
            
            if st.button("Get Similar Models"):
                model_name = f"{selected_brand} {selected_model}"
                recommendations = recommender.get_recommendations_by_model(model_name, num_recommendations)
                
                if isinstance(recommendations, str):
                    st.error(recommendations)
                else:
                    st.success(f"Top {num_recommendations} recommendations similar to {model_name}")
                    
                    # Convert to DataFrame for better display
                    df_recommendations = pd.DataFrame(recommendations)
                    st.dataframe(df_recommendations)
    
    except Exception as e:
        st.error(f"Error loading the recommendation system: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# CHATBOT PAGE
elif st.session_state.page == 'chat':
    st.markdown("<div style='padding: 20px;'>", unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Home", key="back_from_chat"):
        st.session_state.page = 'home'
        st.experimental_rerun()
    
    st.markdown("<div class='chat-section'>", unsafe_allow_html=True)
    st.header("Chat with Our Support Team")
    st.write("Our AI assistant is ready to help you with any questions about electric scooters.")
    
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.components.v1.iframe(
        "https://cdn.botpress.cloud/webchat/v2.2/shareable.html?configUrl=https://files.bpcontent.cloud/2025/03/18/15/20250318152422-PDTBA3BR.json",
        width=500,
        height=600,
        scrolling=False
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)