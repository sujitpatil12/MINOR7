import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

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
        """
        Get recommendations based on user preferences.
        
        Parameters:
        preferences (dict): Dictionary containing user preferences
            - min_range: Minimum range in km
            - max_price: Maximum price in Euros
            - body_style: Preferred body style (optional)
            - min_seats: Minimum number of seats (optional)
            - fast_charge: Whether fast charging is required (optional)
        top_n (int): Number of recommendations to return
        
        Returns:
        list: List of recommended models
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