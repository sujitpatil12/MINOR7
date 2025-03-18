import streamlit as st
import pandas as pd
import numpy as np

# Define the EVRecommendationSystem class in the same file to avoid import issues
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
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        self.features_normalized = self.scaler.fit_transform(self.features)
        
        # Calculate similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
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
        """
        Get recommendations based on user preferences.
        """
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

# Set page title
st.set_page_config(page_title="EV Recommendation System", layout="wide")

# Initialize the recommendation system
@st.cache_resource
def load_recommender():
    return EVRecommendationSystem('ElectricCarData_Clean.csv')

try:
    recommender = load_recommender()
    system_loaded = True
except Exception as e:
    st.error(f"Error loading the recommendation system: {e}")
    system_loaded = False

# Title
st.title("Electric Vehicle Recommendation System")
st.markdown("Find the perfect electric vehicle based on your preferences!")

if system_loaded:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model-based Recommendations", "Preference-based Recommendations", "EV Dataset Overview"])

    if page == "Model-based Recommendations":
        st.header("Find Similar Electric Vehicles")
        st.write("Get recommendations based on a model you already like.")
        
        # Get available brands
        brands = recommender.get_available_brands()
        selected_brand = st.selectbox("Select Brand", brands)
        
        # Get models for selected brand
        brand_models = recommender.df[recommender.df['Brand'].str.strip() == selected_brand]['Model'].str.strip().unique().tolist()
        selected_model = st.selectbox("Select Model", brand_models)
        
        # Number of recommendations
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
        
        if st.button("Get Recommendations"):
            model_name = f"{selected_brand} {selected_model}"
            recommendations = recommender.get_recommendations_by_model(model_name, num_recommendations)
            
            if isinstance(recommendations, str):
                st.error(recommendations)
            else:
                st.success(f"Top {num_recommendations} recommendations similar to {model_name}")
                
                # Convert to DataFrame for better display
                df_recommendations = pd.DataFrame(recommendations)
                st.dataframe(df_recommendations)

    elif page == "Preference-based Recommendations":
        st.header("Find Electric Vehicles Based on Your Preferences")
        
        # Get price range
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

    else:  # Dataset Overview
        st.header("EV Dataset Overview")
        
        # Display the dataset
        st.subheader("Electric Vehicle Dataset")
        st.dataframe(recommender.df)
        
        # Display some statistics
        st.subheader("Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of EVs", len(recommender.df))
            st.metric("Number of Brands", len(recommender.df['Brand'].unique()))
        
        with col2:
            range_stats = recommender.get_range_stats()
            st.metric("Average Range (km)", f"{range_stats['mean']}")
            st.metric("Maximum Range (km)", f"{range_stats['max']}")
        
        with col3:
            price_range = recommender.get_price_range()
            st.metric("Average Price (€)", f"{int(recommender.df['PriceEuro'].mean())}")
            st.metric("Price Range (€)", f"{price_range['min']} - {price_range['max']}")
        
        # Display charts
        st.subheader("Data Visualization")
        
        # Range distribution
        st.write("Range Distribution")
        range_chart = pd.DataFrame(recommender.df['Range_Km'])
        st.bar_chart(range_chart)
        
        # Price vs Range
        st.write("Price vs Range")
        price_range_df = recommender.df[['PriceEuro', 'Range_Km']].rename(columns={'PriceEuro': 'Price (€)', 'Range_Km': 'Range (km)'})
        st.scatter_chart(price_range_df)