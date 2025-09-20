from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def load_and_train_model():
    """Load data, train the best model (Random Forest), and save it"""
    global model, scaler
    
    # Load the dataset
    df_rainfall = pd.read_csv("austin_weather_clean.csv")
    df_rainfall.drop(columns=["DewPointHighF", "DewPointLowF"], inplace=True)
    
    # Remove outliers
    low, high = df_rainfall["VisibilityHighMiles"].quantile([0.01, 1])
    mask_visibilityH = df_rainfall["VisibilityHighMiles"].between(low, high)
    low, high = df_rainfall["SeaLevelPressureAvgInches"].quantile([0.25, 1])
    mask_seaLevel = df_rainfall["SeaLevelPressureAvgInches"].between(low, high)
    
    df_rainfall = df_rainfall[mask_visibilityH & mask_seaLevel]
    
    # Prepare features and target
    input_ds = df_rainfall.drop(columns=["PrecipitationSumInches", "Unnamed: 0"])
    output_ds = df_rainfall["PrecipitationSumInches"]
    
    # Split the data
    input_train, input_test, output_train, output_test = train_test_split(
        input_ds, output_ds, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    input_train_scaled = scaler.fit_transform(input_train)
    input_test_scaled = scaler.transform(input_test)
    
    # Train Random Forest model (best performing model from the notebook)
    model = RandomForestRegressor(random_state=42)
    model.fit(input_train_scaled, output_train)
    
    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model trained and saved successfully!")

def load_saved_model():
    """Load the saved model and scaler"""
    global model, scaler
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print("Model loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        return False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make rainfall prediction"""
    try:
        # Get input data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = [
            data['temp_high_f'],
            data['temp_avg_f'],
            data['temp_low_f'],
            data['dew_point_avg_f'],
            data['humidity_high_percent'],
            data['humidity_avg_percent'],
            data['humidity_low_percent'],
            data['sea_level_pressure_avg_inches'],
            data['visibility_high_miles'],
            data['visibility_avg_miles'],
            data['visibility_low_miles'],
            data['wind_high_mph'],
            data['wind_avg_mph'],
            data['wind_gust_mph']
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Ensure prediction is not negative
        prediction = max(0, prediction)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 3),
            'message': f'Predicted rainfall: {round(prediction, 3)} inches'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model_info')
def model_info():
    """Return model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Random Forest Regressor',
        'features': [
            'TempHighF', 'TempAvgF', 'TempLowF', 'DewPointAvgF',
            'HumidityHighPercent', 'HumidityAvgPercent', 'HumidityLowPercent',
            'SeaLevelPressureAvgInches', 'VisibilityHighMiles', 'VisibilityAvgMiles',
            'VisibilityLowMiles', 'WindHighMPH', 'WindAvgMPH', 'WindGustMPH'
        ],
        'target': 'PrecipitationSumInches'
    })

if __name__ == '__main__':
    # Try to load existing model, if not found, train a new one
    if not load_saved_model():
        load_and_train_model()
    
    # Get port from environment variable (for Render) or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
